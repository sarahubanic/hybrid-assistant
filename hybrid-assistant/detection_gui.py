from duckduckgo_search import DDGS
import cv2
import re
import numpy as np
import pickle
import requests
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import PIL.Image, PIL.ImageTk
import os
import json
import traceback # <-- DODATA ISPRAVKA
from datetime import datetime
try:
    # Optional: CLIP for visual embeddings
    from transformers import CLIPProcessor, CLIPModel
    import torch
except Exception:
    CLIPProcessor = None
    CLIPModel = None
    torch = None
from ultralytics import YOLO
try:
    from llama_cpp import Llama
except Exception:
    # llama_cpp is optional if using Ollama HTTP backend; allow the GUI to run without it
    Llama = None

class DetectionGUI:
    def internet_search(self, query, max_results=3):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                return [r['body'] for r in results]
        except Exception as e:
            return [f"Greška u pretrazi: {e}"]
    def send_with_frame(self):
        """Placeholder for the 'Send + Img' button. Implement sending message with image here."""
        messagebox.showinfo("Send + Img", "This feature is not yet implemented.")

    def start_learning(self):
        """Placeholder for the 'Teach Me Something' button. Implement learning logic here."""
        messagebox.showinfo("Teach Me Something", "Learning mode is not yet implemented.")

    def append_to_chat(self, message):
        """Append a message to the chat display and history"""
        if hasattr(self, 'chat_display'):
            self.chat_display.configure(state=tk.NORMAL)
            self.chat_display.insert(tk.END, message + "\n\n")
            self.chat_display.see(tk.END)
            self.chat_display.configure(state=tk.DISABLED)
        if hasattr(self, 'chat_history'):
            self.chat_history.append(message)
            self.save_chat_history()
            
    def __init__(self, window, window_title):
        self.is_camera_on = False  # Ensure this is always set first
        self.window = window
        self.window.title(window_title)
        
        # --- IZMENA (Zahtev 1): Omogućavanje rastezanja glavnog prozora ---
        # Ovo je bilo pomenuto kao već omogućeno, potvrđujemo
        self.window.resizable(True, True)
        
        # Initialize paths
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_path, "..", "models")
        self.learning_dir = os.path.join(self.base_path, "..", "learned_items")
        self.knowledge_file = os.path.join(self.learning_dir, "knowledge.json")
        self.chat_history_file = os.path.join(self.learning_dir, "chat_history.json")
        
        # Create necessary directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)
        # Enable auto-describe for practical mode
        self.auto_describe = True
        # Backend selection: default to 'ollama' (lightweight)
        self.backend = os.environ.get('HA_BACKEND', 'ollama')
        self.current_detections = []
        self.knowledge_base = self.load_knowledge_base()
        self.chat_history = self.load_chat_history()
        self.is_learning_mode = False
        # Do not auto-describe every frame; use a button to trigger description
        self.auto_describe = False
        # Backend selection: default to 'ollama' (lightweight)
        self.backend = os.environ.get('HA_BACKEND', 'ollama')
        # Embedding model (lightweight, for retrieval)
        self.embed_model = os.environ.get('HA_EMBED_MODEL', 'nomic-embed-text:latest')
        # Chat model (for interpretation and responses) - prefer gemma3:1b if available
        self.ollama_model = os.environ.get('HA_CHAT_MODEL', 'gemma3:1b')
        
        # --- IZMENA (Zahtev 2): Uklanjanje ORB podešavanja iz init ---
        # Uklonjena podešavanja za ORB (orb_features, match_threshold, match_distance_cutoff)
        # jer prelazimo isključivo na CLIP.
        
        # Embedding matcher threshold (cosine)
        try:
            self.visual_embed_threshold = float(os.environ.get('HA_VIS_EMBED_THRESH', '0.32'))
        except Exception:
            self.visual_embed_threshold = 0.32
        # Available Ollama models (will be queried at startup)
        self.available_models = []

        # --- IZMENA (Zahtev 2): Putanje za CLIP podatke ---
        self.visual_embeddings = {} # label -> avg_vector (NE KORISTI SE TRENUTNO)
        self.visual_embeddings_samples = [] # Lista {'label': ..., 'embedding': ...}
        self.clip_processor = None
        self.clip_model = None
        
        # Definišemo putanje za CLIP JSON fajlove
        self.clip_samples_file = os.path.join(self.learning_dir, 'visual_embeddings_samples.json')
        self.clip_avg_file = os.path.join(self.learning_dir, 'visual_embeddings.json')
        
        # Create GUI elements - this must be done in __init__
        self.create_gui()
        # State for add-object rectangle drawing
        self.add_object_mode = False 
        self._rect_start = None
        self._rect_id = None
        self.last_frame = None
        # --- IZMENA: Dodajemo novi state za sinhronizaciju frejmova ---
        self.last_raw_frame_processed = None # Sirovi frejm koji je poslat na obradu
        # --- KRAJ IZMENE ---
        # State for freeze + relabel mode
        self.is_frozen = False
        self.frozen_frame = None
        self.frozen_detections = []
        self.correction_pending = None
        
        # --- NEW STATE VARS for Teach Me Dialog ---
        self.dialog_canvas_widget = None
        self.dialog_is_frozen = False
        self.dialog_frozen_frame = None
        self.dialog_update_running = False
        # Track greeted persons for session
        self.greeted_persons = set()
        
        self.dialog_preview_canvas = None
        self.dialog_preview_photo = None
        
        # --- IZMENA (HIBRIDNI MOD): Preimenovano ---
        self.pending_dialog_crop = None # Sadrži isečak iz Rectangle/Polygon
        self.dialog_context_var = None # Referenca na radio dugmad
        # --- KRAJ IZMENE ---
        
        self.polygon_points = []
        self.polygon_draw_id = None
        self._temp_poly_line = None
        self.current_drawing_mode = None
        
        # Bind global keyboard events
        self.window.bind('<Control-f>', self._on_freeze_key)
        self.window.bind('<Control-F>', self._on_freeze_key)
        self.window.bind('<Escape>', self._on_escape_key)
        
    def load_knowledge_base(self):
        """Load existing knowledge base from JSON file"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
                    self.last_processed_frame = None
                    if '_corrections' not in kb:
                        kb['_corrections'] = {}
                    return kb
            return {'_corrections': {}}
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return {'_corrections': {}}
            
    def load_chat_history(self):
        """Load existing chat history from JSON file"""
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return []
            
    def save_chat_history(self):
        """Save chat history to JSON file"""
        try:
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {e}")
            
    def save_knowledge_base(self):
        """Save knowledge base to JSON file"""
        try:
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] Knowledge base saved successfully with {len(self.knowledge_base)} items")
            for name, data in self.knowledge_base.items():
                context = data.get('context', 'unknown')
                desc = data.get('description', '')[:50]
                print(f"  - {name}: {desc}... (context: {context})")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def get_permanent_knowledge(self):
        """Vraća listu stringova sa svim 'General Knowledge' pravilima."""
        rules = []
        try:
            for name, info in self.knowledge_base.items():
                if info.get('context') == 'general':
                    rule_text = f"{name}: {info.get('description', '')}"
                    if rule_text.strip() and rule_text != ": ":
                        rules.append(rule_text)
        except Exception as e:
            print(f"[PERMA_KNOWLEDGE] Error reading general rules: {e}")
        return rules
            
    # ----- Face & Visual DB utilities -----
    def load_face_recognizer(self):
        """Load LBPH face recognizer, label mapping, and master sample list if present"""
        self.face_model_file = os.path.join(self.learning_dir, "face_recognizer.yml")
        self.face_labels_file = os.path.join(self.learning_dir, "face_labels.pkl")
        self.face_samples_file = os.path.join(self.learning_dir, "face_samples_master.pkl")
        
        try:
            if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            else:
                print("Warning: OpenCV face module (cv2.face) not available. Face recognition disabled.")
                self.face_recognizer = None

            if self.face_recognizer and os.path.exists(self.face_model_file):
                self.face_recognizer.read(self.face_model_file)
                print(f"Loaded trained face model from {self.face_model_file}")

            if os.path.exists(self.face_labels_file):
                with open(self.face_labels_file, 'rb') as f:
                    self.face_labels = pickle.load(f)
            else:
                self.face_labels = {}
            
            self.face_labels_rev = {v: k for k, v in self.face_labels.items()}

            if os.path.exists(self.face_samples_file):
                with open(self.face_samples_file, 'rb') as f:
                    data = pickle.load(f)
                    self.all_face_images = data.get('images', [])
                    self.all_face_labels = data.get('labels', [])
                print(f"Loaded {len(self.all_face_images)} master face samples.")
            else:
                self.all_face_images = []
                self.all_face_labels = []

        except Exception as e:
            print(f"Error initializing face recognizer: {e}")
            self.face_recognizer = None
            self.face_labels = {}
            self.face_labels_rev = {}
            self.all_face_images = []
            self.all_face_labels = []
    
    def save_face_recognizer(self):
        """Saves the trained .yml model and the label mapping .pkl file"""
        try:
            if self.face_recognizer is not None:
                print(f"Saving face model to {self.face_model_file}")
                self.face_recognizer.write(self.face_model_file)
            
            with open(self.face_labels_file, 'wb') as f:
                print(f"Saving face labels to {self.face_labels_file}")
                pickle.dump(self.face_labels, f)
        except Exception as e:
            print(f"Error saving face recognizer: {e}")
    
    def add_face_sample(self, name, face_gray):
        """Add a face sample, retrain the model, and save everything."""
        try:
            if self.face_recognizer is None:
                if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
                    print("Creating new LBPH face recognizer")
                    self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                else:
                    print("Cannot create LBPHFaceRecognizer: cv2.face missing. Skipping face training.")
                    return False
            
            print(f"\nAdding face sample for {name}")
            
            if name not in self.face_labels:
                new_id = max(self.face_labels.values(), default=-1) + 1
                self.face_labels[name] = new_id
                self.face_labels_rev[new_id] = name
                print(f"Created new face ID {new_id} for {name}")
            
            label_id = self.face_labels[name]
            
            self.all_face_images.append(face_gray)
            self.all_face_labels.append(label_id)
            print(f"Appended new sample. Total samples: {len(self.all_face_images)}")
            
            try:
                with open(self.face_samples_file, 'wb') as f:
                    data = {'images': self.all_face_images, 'labels': self.all_face_labels}
                    pickle.dump(data, f)
                print(f"Saved master samples to {self.face_samples_file}")
            except Exception as e:
                print(f"Error saving master samples file: {e}")
            
            if self.all_face_images and self.all_face_labels:
                print(f"Training face recognizer with {len(self.all_face_images)} total samples...")
                self.face_recognizer.train(self.all_face_images, np.array(self.all_face_labels))
                print("Training complete.")
                
                self.save_face_recognizer()
                print("Face recognizer trained and saved successfully.")
                return True
            else:
                print("Warning: No face samples available for training.")
                return False
        except Exception as e:
            print(f"Error adding face sample: {e}")
            return False
    
    def predict_face(self, face_gray):
        """Predict a face name from a grayscale face crop. Returns (name, confidence) or (None, None)"""
        try:
            if self.face_recognizer is None or not self.face_labels_rev:
                return None, None
            label, conf = self.face_recognizer.predict(face_gray)
            name = self.face_labels_rev.get(label)
            return name, conf
        except Exception as e:
            return None, None
    
    # --- IZMENA (Zahtev 2): Ažurirano učitavanje baze podataka ---
    def load_visual_db(self):
        """Load Visual DB. Sada učitava samo CLIP embeddinge."""
        
        # 1. Učitaj CLIP embeddinge (glavni mehanizam)
        self.load_clip_embeddings()
        
        # 2. Inicijalizuj CLIP model ako je dostupan
        if CLIPModel is not None and torch is not None:
             self._init_clip_model() # Inicijalizuj model
        else:
            print("[CLIP] Transformers/Torch nije dostupan. Vizuelno prepoznavanje je onemogućeno.")
            
        # 3. ORB kod je uklonjen/onemogućen
        self.visual_db_file = os.path.join(self.learning_dir, "visual_db.pkl")
        self.visual_db = {} # ORB baza se više ne koristi
        self.orb = None
        self.bf = None
        print(f"[VISUAL_DB] Učitano {len(self.visual_embeddings_samples)} CLIP uzoraka. ORB je onemogućen.")

    def load_clip_embeddings(self):
        """Učitava CLIP embeddinge (uzorke i proseke) iz JSON fajlova."""
        try:
            if os.path.exists(self.clip_avg_file):
                with open(self.clip_avg_file, 'r', encoding='utf-8') as f:
                    self.visual_embeddings = json.load(f)
                print(f"Loaded visual embeddings (avg) for {len(self.visual_embeddings)} labels")
                
            if os.path.exists(self.clip_samples_file):
                with open(self.clip_samples_file, 'r', encoding='utf-8') as f:
                    self.visual_embeddings_samples = json.load(f)
                print(f"Loaded {len(self.visual_embeddings_samples)} per-sample embeddings")
                
        except Exception as e:
            print(f"Error loading visual embeddings: {e}")
            self.visual_embeddings = {}
            self.visual_embeddings_samples = []

    def save_clip_embeddings(self):
        """Čuva bazu CLIP embeddinga (uzoraka) u JSON fajl."""
        try:
            # Trenutno čuvamo samo sirove uzorke
            with open(self.clip_samples_file, 'w', encoding='utf-8') as f:
                json.dump(self.visual_embeddings_samples, f, indent=2)
            print(f"Saved {len(self.visual_embeddings_samples)} CLIP samples to {self.clip_samples_file}")
            
            # TODO: Opciono implementirati preračunavanje proseka (self.visual_embeddings)
            
        except Exception as e:
            print(f"Error saving clip embeddings: {e}")

    # --- IZMENA (Zahtev 2): Uklonjena 'save_visual_db' (ORB) ---
    # --- IZMENA (Zahtev 2): Uklonjena 'add_visual_sample' (ORB) ---

    def _init_clip_model(self):
        """Pomoćna funkcija za inicijalizaciju CLIP modela."""
        if self.clip_processor is not None and self.clip_model is not None:
            return True # Već inicijalizovan
            
        if CLIPModel is None or torch is None:
            print("[CLIP] Transformers/Torch nije dostupno.")
            return False
            
        try:
            print("[CLIP] Učitavanje CLIP modela (clip-vit-base-patch32)...")
            self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', use_fast=True)
            self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            if torch is not None:
                self.clip_model = self.clip_model.to('cpu') # Koristi CPU za sada
            print("[CLIP] CLIP model uspešno učitan.")
            return True
        except Exception as e:
            print(f"[CLIP] Neuspešna inicijalizacija CLIP modela: {e}")
            self.clip_processor = None
            self.clip_model = None
            return False

    def generate_clip_embedding(self, img_bgr):
        """Generiše CLIP embedding za datu BGR sliku. Vraća numpy vektor ili None."""
        if not self._init_clip_model(): # Osiguraj da je model učitan
            return None
            
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(img_rgb)
            
            inputs = self.clip_processor(images=pil_image, return_tensors='pt')
            with torch.no_grad():
                qf = self.clip_model.get_image_features(**inputs)
                
            qf = qf / qf.norm(p=2, dim=-1, keepdim=True)
            qv = qf[0].cpu().numpy()
            return qv
            
        except Exception as e:
            print(f"[CLIP_EMBED] Greška pri generisanju embeddinga: {e}")
            return None

    def add_clip_embedding(self, name, description, img_bgr):
        """
        Generiše CLIP embedding, dodaje ga u bazu i čuva u knowledge.json.
        Ovo je nova centralna funkcija za vizuelno učenje.
        """
        
        # 1. Generiši embedding
        embedding = self.generate_clip_embedding(img_bgr)
        
        if embedding is None:
            print(f"[TEACH] Neuspešno generisanje CLIP embeddinga za {name}.")
            return False
            
        # 2. Dodaj u listu uzoraka
        new_sample = {
            'label': name,
            'description': description,
            'embedding': embedding.tolist(), # Sačuvaj kao listu
            'timestamp': datetime.now().isoformat()
        }
        self.visual_embeddings_samples.append(new_sample)
        
        # 3. Sačuvaj bazu embeddinga (JSON)
        self.save_clip_embeddings()
        
        # 4. Sačuvaj i u glavnu bazu znanja (knowledge.json)
        if name not in self.knowledge_base:
            self.knowledge_base[name] = {
                "description": description,
                "examples": [],
                "context": "visual", # Postavi kontekst na 'visual'
                "learned_on": datetime.now().isoformat()
            }
        else:
            # Ažuriraj opis ako je dat
            if description:
                self.knowledge_base[name]['description'] = description
            self.knowledge_base[name]['context'] = 'visual' # Osiguraj kontekst
            
        # Dodaj informaciju o vizuelnom uzorku
        visual_data_info = {
            "capture_time": datetime.now().isoformat(),
            "embedding_stored": True,
            "source": "clip_teach"
        }
        self.knowledge_base[name].setdefault('visual_data_list', []).append(visual_data_info)
        self.knowledge_base[name]['visual_data'] = visual_data_info # Za kompatibilnost sa starim
        
        self.save_knowledge_base()
        
        print(f"[TEACH] Uspešno dodat CLIP embedding za '{name}'. Ukupno uzoraka: {len(self.visual_embeddings_samples)}")
        return True

    # --- IZMENA (Zahtev 2): 'match_visual' sada koristi isključivo CLIP ---
    def match_visual(self, img_bgr, return_score=False):
        """
        Pokušava da upari sliku sa poznatim vizuelnim uzorcima koristeći isključivo CLIP.
        Vraća ime ili (ime, skor) ako je return_score=True.
        """
        try:
            # 1. Proveri da li je CLIP dostupan
            if not self.visual_embeddings_samples or CLIPModel is None or torch is None:
                if return_score:
                    return None, 0.0
                return None
                
            # 2. Generiši embedding za upitnu sliku
            qv = self.generate_clip_embedding(img_bgr)
            if qv is None:
                if not self.dialog_update_running:
                    print("[EMBED_MATCH] Neuspešno generisanje query embeddinga.")
                if return_score:
                    return None, 0.0
                return None
                
            # 3. Normalizuj query vektor
            qv_norm = qv / (np.linalg.norm(qv) + 1e-12)
            
            best_name = None
            best_score = -1.0
            
            # 4. Uporedi sa svim sačuvanim uzorcima
            # Koristimo self.visual_embeddings_samples
            for s in self.visual_embeddings_samples:
                try:
                    emb = np.array(s.get('embedding'))
                    if emb.ndim != 1:
                        continue
                        
                    # 5. Kosinusna sličnost (dot proizvod normalizovanih vektora)
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
                    sim = float(np.dot(qv_norm, emb_norm))
                    
                    if sim > best_score:
                        best_score = sim
                        best_name = s.get('label')
                except Exception:
                    continue
            
            if not self.dialog_update_running:
                print(f"[EMBED_MATCH] (CLIP-Only) best={best_name} score={best_score:.3f} thr={self.visual_embed_threshold}")
            
            # 6. Vrati rezultat ako je iznad praga
            if best_name and best_score >= float(self.visual_embed_threshold):
                if return_score:
                    return best_name, float(best_score)
                return best_name
                
        except Exception as e:
            print(f"[EMBED] (CLIP-Only) Matching failed: {e}")

        # Ako ORB fallback nije dozvoljen (Zahtev 2), vraćamo None
        if return_score:
            return None, 0.0
        return None
    
    def apply_correction(self, detected_label):
        """Apply correction map: if detected_label is a known false positive, remap it"""
        try:
            corrections = self.knowledge_base.get('_corrections', {})
            if detected_label in corrections:
                corrected = corrections[detected_label]
                print(f"[CORRECTION] {detected_label} -> {corrected}")
                return corrected
        except Exception:
            pass
        return detected_label
    
    def _on_freeze_key(self, event):
        """Handler for Ctrl+F: freeze current frame for manual relabeling"""
        if not self.is_camera_on:
            return
        if self.is_frozen:
            self.is_frozen = False
            self.frozen_frame = None
            self.frozen_detections = []
            print("[CORRECTION] Unfroze frame. Returning to live detection.")
            self.append_to_chat("System: Frame unfrozen. Returning to live mode.\n")
        else:
            if self.last_frame is not None:
                self.is_frozen = True
                self.frozen_frame = self.last_frame.copy()
                self.frozen_detections = list(self.current_detections)
                print(f"[CORRECTION] Frame frozen with detections: {self.frozen_detections}")
                self.append_to_chat(f"System: Frame frozen. Current detections: {', '.join(self.frozen_detections)}\n")
                self.append_to_chat("System: Type in the chat: \"Correct: [wrong] to [correct]\" to create a correction mapping.\n")
    
    def _on_escape_key(self, event):
        """Handler for Escape: unfreeze if frozen"""
        if self.is_frozen:
            self._on_freeze_key(None)
    
    def handle_correction_command(self, message):
        """Parse and execute correction commands like 'Correct: toothbrush to cigara'"""
        import re
        match = re.match(r'Correct:\s*(\S+)\s+to\s+(\S+)', message.strip(), re.IGNORECASE)
        if match:
            wrong_label, correct_label = match.group(1), match.group(2)
            if correct_label not in self.knowledge_base:
                self.append_to_chat(f"System: Label '{correct_label}' not found in knowledge base. Available: {', '.join([k for k in self.knowledge_base.keys() if k != '_corrections'])}\n")
                return False
            self.knowledge_base['_corrections'][wrong_label] = correct_label
            self.save_knowledge_base()
            print(f"[CORRECTION] Added mapping: {wrong_label} -> {correct_label}")
            self.append_to_chat(f"System: Correction saved! From now on, when the model detects '{wrong_label}', it will be relabeled as '{correct_label}'.\n")
            return True
        return False
            
    def __post_init__(self):
        """Initialize models and GUI after basic setup"""
        self.load_models()
        self.create_gui()
        self.delay = 15
        self.update()
        
        if hasattr(self, 'chat_display'):
            self.append_to_chat("Assistant: Hello! I'm your AI assistant. You can chat with me anytime, and when you turn on the camera, I'll also tell you what I see.")
        
    def load_models(self):
        self.llm = None
        print("[CONFIG] Using Ollama backend (local LLM disabled for lightweight operation)")
        
        try:
            url = os.environ.get('OLLAMA_MODELS_URL', 'http://127.0.0.1:11434/api/tags')
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self.available_models = [m['name'] for m in data.get('models', [])]
                print(f"[OLLAMA] Available models: {self.available_models}")
                if 'gemma3:1b' in self.available_models:
                    self.ollama_model = os.environ.get('HA_CHAT_MODEL', 'gemma3:1b')
                else:
                    if self.ollama_model not in self.available_models and len(self.available_models) > 0:
                        self.ollama_model = self.available_models[0]
            else:
                print(f"[OLLAMA] Warning: Could not fetch models (status {resp.status_code})")
        except Exception as e:
            print(f"[OLLAMA] Warning: Could not query models: {e}")
        
        try:
            yolo_path = os.path.join(self.models_dir, "yolov8n.pt")
            self.yolo_model = YOLO(yolo_path)
            print("YOLO model initialized successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None
        
        try:
            opencv_home = os.path.dirname(cv2.__file__)
            cascade_path = os.path.join(opencv_home, 'data', 'haarcascade_frontalface_default.xml')
            print(f"Looking for face cascade at: {cascade_path}")
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    print("Warning: Face cascade loaded but is empty")
                    self.face_cascade = None
                else:
                    print("Face detection cascade loaded successfully")
            else:
                print(f"Error: Face cascade file not found at {cascade_path}")
                self.face_cascade = None
        except Exception as e:
            print(f"Error initializing face cascade: {e}")
            self.face_cascade = None

        self.load_face_recognizer()
        self.load_visual_db() # Ovo sada učitava CLIP
    
    def ollama_generate(self, prompt, model=None, max_tokens=256, temperature=0.2):
        """Call local Ollama HTTP API to generate text. Returns raw text on success."""
        url = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
        if model is None:
            model = self.ollama_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                error_msg = resp.text[:200] if resp.text else "Unknown error"
                print(f"[OLLAMA] HTTP error {resp.status_code}: {error_msg}")
                if "CUDA error" in error_msg or "runner process has terminated" in error_msg:
                    print("[OLLAMA] CUDA error detected - Ollama runner crashed. Restarting may be needed.")
                    return f"[Ollama offline - CUDA crash]"
                return f"[OLLAMA ERROR] Status {resp.status_code}"
            
            text_parts = []
            for line in resp.text.strip().split('\n'):
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if 'response' in j:
                        text_parts.append(j['response'])
                    if j.get('done', False):
                        break
                except Exception as e:
                    print(f"[OLLAMA] Error parsing JSON line: {e}")
            
            if text_parts:
                return ''.join(text_parts)
            return "[OLLAMA] No response text found"
        except Exception as e:
            print("[OLLAMA] generate failed:", e)
            if "Connection" in str(e) or "timeout" in str(e).lower():
                return "[Ollama not responding - check if running]"
            return f"[OLLAMA ERROR] {e}"
    
    def ollama_embed(self, text, model=None):
        """Call local Ollama HTTP API embedding endpoint. Returns a vector or None on error."""
        url = os.environ.get('OLLAMA_EMBED_URL', 'http://127.0.0.1:11434/api/embed')
        model = model or os.environ.get('HA_EMBED_MODEL', 'nomic-embed-text:latest')
        payload = {
            "model": model,
            "input": text
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            j = resp.json()
            if isinstance(j, dict) and 'embedding' in j:
                return j['embedding']
            if isinstance(j, dict) and 'embeddings' in j and len(j['embeddings']) > 0:
                return j['embeddings'][0]
            if isinstance(j, dict) and 'error' in j:
                error_msg = j['error']
                print(f"[OLLAMA EMBED] Unexpected response: {error_msg}")
                if "CUDA error" in error_msg or "runner process has terminated" in error_msg:
                    print("[OLLAMA EMBED] CUDA error - Ollama runner crashed")
                return None
            print(f"[OLLAMA EMBED] Unexpected response: {j}")
            return None
        except Exception as e:
            print("[OLLAMA EMBED] failed:", e)
            if "Connection" in str(e):
                print("[OLLAMA EMBED] Connection failed - Ollama may not be running")
            return None
    
    def retrieve_relevant_knowledge_semantic(self, query_text, top_k=3):
        """Retrieve top-k most similar knowledge entries using embeddings (nomic-embed-text)."""
        if getattr(self, 'backend', 'local') != 'ollama':
            return []
        query_vec = self.ollama_embed(query_text, model=self.embed_model)
        if query_vec is None:
            return []
        
        embeddings_file = os.path.join(self.learning_dir, "knowledge_embeddings.json")
        knowledge_embeddings = {}
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'r') as f:
                    knowledge_embeddings = json.load(f)
            except:
                pass
        
        for name, info in self.knowledge_base.items():
            if name not in knowledge_embeddings:
                desc = info.get('description', '')
                vec = self.ollama_embed(desc, model=self.embed_model)
                if vec:
                    knowledge_embeddings[name] = vec
        
        try:
            with open(embeddings_file, 'w') as f:
                json.dump(knowledge_embeddings, f)
        except:
            pass
        
        def cosine_sim(v1, v2):
            import math
            dot = sum(a*b for a,b in zip(v1, v2))
            norm1 = math.sqrt(sum(a*a for a in v1))
            norm2 = math.sqrt(sum(a*a for a in v2))
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot / (norm1 * norm2)
        
        scores = []
        for name, vec in knowledge_embeddings.items():
            if name in self.knowledge_base:
                if self.knowledge_base[name].get('context') == 'general':
                    continue
                sim = cosine_sim(query_vec, vec)
                scores.append((name, sim, self.knowledge_base[name]))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def create_gui(self):
        main_container = ttk.Frame(self.window)
        # --- IZMENA (Zahtev 1): Glavni kontejner se rasteže ---
        main_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_container)
        # --- IZMENA (Zahtev 1): Levi frejm se rasteže ---
        left_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(left_frame, width=800, height=600)
        # --- IZMENA (Zahtev 1): Canvas se rasteže ---
        self.canvas.pack(pady=5, fill=tk.BOTH, expand=True)
        
        controls = ttk.Frame(left_frame)
        controls.pack(fill=tk.X, pady=5)
        
        self.cam_btn = ttk.Button(controls, text="Turn Camera On", command=self.toggle_camera)
        self.cam_btn.pack(side=tk.LEFT, padx=5)
        
        self.learn_btn = ttk.Button(controls, text="Teach Me Something", command=self.start_learning)
        self.learn_btn.pack(side=tk.LEFT, padx=5)
        
        self.describe_btn = ttk.Button(controls, text="Describe Scene", command=self.describe_scene)
        self.describe_btn.pack(side=tk.LEFT, padx=5)

        embed_lbl = ttk.Label(controls, text="Embed:")
        embed_lbl.pack(side=tk.LEFT, padx=(10,2))
        self.embed_model_var = tk.StringVar(value=self.embed_model)
        embed_combo = ttk.Combobox(
            controls, 
            textvariable=self.embed_model_var, 
            values=self.available_models if self.available_models else ['nomic-embed-text:latest'],
            width=20
        )
        embed_combo.pack(side=tk.LEFT)
        embed_combo.bind('<<ComboboxSelected>>', lambda e: setattr(self, 'embed_model', self.embed_model_var.get()))
        
        chat_lbl = ttk.Label(controls, text="Chat:")
        chat_lbl.pack(side=tk.LEFT, padx=(10,2))
        self.ollama_model_var = tk.StringVar(value=self.ollama_model)
        chat_combo = ttk.Combobox(
            controls, 
            textvariable=self.ollama_model_var, 
            values=self.available_models if self.available_models else ['gemma3:1b','qwen2.5-coder:1.5b-base','mistral:latest'],
            width=20
        )
        chat_combo.pack(side=tk.LEFT)
        def _on_chat_model_change(e=None):
            val = self.ollama_model_var.get()
            setattr(self, 'ollama_model', val)
            os.environ['HA_CHAT_MODEL'] = val
            print(f"[CONFIG] Chat model set to: {val}")
        chat_combo.bind('<<ComboboxSelected>>', _on_chat_model_change)

        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        chat_frame = ttk.LabelFrame(right_frame, text="Chat")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=50, height=30)
        self.chat_display.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.message_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=40, height=3)
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        send_btn = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_btn.pack(side=tk.RIGHT)
        send_img_btn = ttk.Button(input_frame, text="Send + Img", command=self.send_with_frame)
        send_img_btn.pack(side=tk.RIGHT, padx=(0,5))
        
        self.message_input.bind("<Return>", lambda e: self.send_message() if not e.state & 0x1 else None)
        
        self.status = ttk.Label(right_frame, text="Ready")
        self.status.pack(pady=5)
    
    def find_relevant_knowledge(self, message):
        """Find relevant knowledge based on user message."""
        relevant_items = []
        
        if getattr(self, 'backend', 'local') == 'ollama':
            semantic_results = self.retrieve_relevant_knowledge_semantic(message, top_k=3)
            for item in semantic_results:
                relevant_items.append({
                    "name": item[0],
                    "info": item[2],
                    "similarity": item[1]
                })
            if relevant_items:
                return relevant_items
        
        message_lower = message.lower()
        for name, info in self.knowledge_base.items():
            if info.get('context') == 'general':
                continue
                
            if (name.lower() in message_lower or 
                any(word.lower() in message_lower for word in info.get("description", "").split())):
                relevant_items.append({
                    "name": name,
                    "info": info
                })
        
        return relevant_items

    def send_message(self):
        print("[DEBUG] send_message() called")
        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            print("[DEBUG] Empty message, returning")
            return
            
        self.message_input.delete("1.0", tk.END)
        self.append_to_chat(f"You: {message}")
        print("[DEBUG] Message appended to chat display")

        # DuckDuckGo search for 'search:' or 'pretrazi:'
        if message.lower().startswith('search:') or message.lower().startswith('pretrazi:'):
            query = message.split(':', 1)[1].strip()
            results = self.internet_search(query)
            self.append_to_chat("Internet rezultati:\n" + "\n".join(results))
            return
        try:
            handled = self.handle_correction_command(message)
            if handled:
                return # Komanda za korekciju je obrađena
        except Exception as e:
            print(f"[CORRECTION] Error in correction handler: {e}")
        
        # --- IZMENA (Zahtev 3): Poziv 'handle_teach_message' ---
        try:
            # handle_teach_message sada vraća True ako je komanda obrađena (uspešno ili ne)
            handled = self.handle_teach_message(message)
            if handled:
                print(f"[DEBUG] 'Teach' command handled, stopping send_message.")
                return 
        except Exception as e:
            print(f"[TEACH] Error in teach handler: {e}")
        # --- KRAJ IZMENE ---
        
        print(f"[DEBUG] Sending message (not a teach command): {message[:50]}...")
        self.attached_image_path = None

        context_parts = []
        print("[DEBUG] Building context...")

        permanent_rules = self.get_permanent_knowledge()
        if permanent_rules:
            context_parts.append("Permanent Instructions (Must Follow):")
            context_parts.extend([f"- {rule}" for rule in permanent_rules])
            context_parts.append("\n")
            print(f"[DEBUG] Added {len(permanent_rules)} permanent rules to context.")

        if self.is_camera_on and self.current_detections:
            context_parts.append(f"Currently seeing: {', '.join(self.current_detections)}")
            try:
                recognized_people = [d for d in self.current_detections if d in self.knowledge_base and self.knowledge_base[d].get('context') == 'face']
                if recognized_people:
                    context_parts.append(f"Recognized people present: {', '.join(recognized_people)}")
                    for person in recognized_people:
                        info = self.knowledge_base.get(person, {}).get('description', '')
                        if info:
                            context_parts.append(f"About {person}: {info}")
            except Exception as e:
                print(f"[DEBUG] Error while adding recognized people to context: {e}")
        print(f"[DEBUG] Camera on: {self.is_camera_on}, Detections: {self.current_detections}")

        relevant_knowledge = self.find_relevant_knowledge(message)
        if relevant_knowledge:
            context_parts.append("Relevant knowledge I have learned (for this question):")
            for item in relevant_knowledge:
                context_parts.append(f"- {item['name']}: {item['info']['description']}")
                if 'examples' in item['info'] and item['info']['examples']:
                    context_parts.append("  Examples: " + ", ".join(item['info']['examples']))
        print("[DEBUG] Knowledge added")

        recent_history = [msg for msg in self.chat_history[-10:] if not msg.startswith("System:")]
        if recent_history:
            context_parts.append("\nRecent conversation:")
            context_parts.extend(recent_history)
        print("[DEBUG] Chat history added")

        context = "\n".join(context_parts)

        try:
            print("[DEBUG] About to call LLM.generate()...")
            prompt = f"""Context:
{context}

Current user message: {message}

Instructions:
Respond as the assistant based on all context provided.

Assistant:"""

            print("[DEBUG] Prompt prepared, about to call model backend...")
            print(f"[DEBUG] Full prompt length: {len(prompt)} chars")

            if getattr(self, 'backend', 'local') == 'ollama':
                print(f"[DEBUG] Using Ollama backend with model={self.ollama_model}")
                text = self.ollama_generate(prompt + "\nAssistant: ", model=self.ollama_model, max_tokens=400)
                if text:
                    self.append_to_chat(f"Assistant: {text}")
                else:
                    self.append_to_chat("Assistant: (No response generated - Ollama returned empty)")
            else:
                if not self.llm:
                    messagebox.showerror("Error", "LLM not initialized")
                    return
                response = self.llm(
                    prompt + "\nAssistant: ",
                    max_tokens=400,
                    stop=["\nUser:", "User:", "<end_of_turn>"]
                )
                print(f"[DEBUG] LLM returned: {response is not None}")
                try:
                    if response and 'choices' in response and len(response['choices']) > 0:
                        text = response['choices'][0]['text'].strip()
                        if text:
                            self.append_to_chat(f"Assistant: {text}")
                        else:
                            self.append_to_chat("Assistant: (No response generated - possible empty stop token hit)")
                    else:
                        self.append_to_chat("Assistant: (Response structure invalid)")
                except Exception as e:
                    print("[DEBUG] Error parsing LLM response:", e)
                    self.append_to_chat(f"Assistant: (Error parsing response: {e})")
        except Exception as e:
            # import traceback # <-- Ovo je sada na vrhu fajla
            print(f"[DEBUG] Exception in send_message: {e}")
            print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            self.append_to_chat(f"Assistant: Sorry, I encountered an error: {str(e)}")
        finally:
            self.attached_image_path = None

    def describe_scene(self):
        """Generate a single scene description on demand using current detections (Ollama backend)."""
        detected = list(self.current_detections) if self.current_detections else []
        print(f"[DESCRIBE] Describe button pressed. Current detections: {detected}")

        if not self.is_camera_on:
            self.append_to_chat("Assistant: I cannot see the camera right now.")
            return

        recognized_people = [d for d in detected if d in self.knowledge_base and self.knowledge_base[d].get('context') == 'face']
        objects = [d for d in detected if d not in recognized_people and d != 'Unknown Person']
        unknown_people = [d for d in detected if d == 'Unknown Person']

        description_parts = []
        if recognized_people:
            description_parts.append(f"I see {', '.join(recognized_people)}.")
            for person in recognized_people:
                info = self.knowledge_base.get(person, {}).get('description', '')
                if info:
                    description_parts.append(f"About {person}: {info}")
        if unknown_people:
            description_parts.append("I also see an unknown person.")
        if objects:
            description_parts.append(f"I also see: {', '.join(objects)}.")

        if not description_parts:
            description_prompt = "I don't see anything notable right now."
        else:
            description_prompt = " ".join(description_parts)

        try:
            print(f"[DESCRIBE] Prompt: {description_prompt}")
            
            permanent_rules = self.get_permanent_knowledge()
            context_parts = []
            if permanent_rules:
                context_parts.append("Permanent Instructions (Must Follow):")
                context_parts.extend([f"- {rule}" for rule in permanent_rules])
                context_parts.append("\n")
            
            context_parts.append(f"Scene context: {description_prompt}")
            context_parts.append("Instruction: Describe this scene based *only* on the context. Do not invent actions or emotions.")
            
            model_prompt = "\n".join(context_parts)

            text = self.ollama_generate(model_prompt, model=self.ollama_model, max_tokens=300)
            if text and not text.startswith("[OLLAMA"):
                self.append_to_chat(f"Assistant: {text}")
                print(f"[DESCRIBE] Response: {text[:120]}...")
            else:
                self.append_to_chat(f"Assistant: (Failed to generate description: {text})")
        except Exception as e:
            print(f"[DESCRIBE] Error generating scene description: {e}")
            self.append_to_chat(f"Assistant: Error describing scene: {e}")

    # --- IZMENA (Zahtev 3): Ažurirana 'handle_teach_message' ---
    def handle_teach_message(self, message: str) -> bool:
        """
        Parsira 'Teach:' komande.
        - 'Teach: <label> - <description>' (NOVO): Pokreće vizuelno učenje sa kamere.
        - 'Teach: name=...; desc=...' (STARO): Zadržano za tekstualno učenje.
        Vraća True ako je poruka prepoznata kao 'Teach' komanda (da se zaustavi slanje LLM-u).
        """
        
        # 1. Proveri da li je uopšte 'Teach' komanda
        m = re.match(r"^\s*teach\s*[:]?\s*(.+)$", message, re.IGNORECASE)
        if not m:
            return False # Nije 'Teach' komanda, nastavi sa send_message

        body = m.group(1).strip()
        
        # 2. PROVERI PRVO NOVI FORMAT (Zahtev 3): 'Teach: label - description'
        m2 = re.match(r"^([^\-—:]+)\s*[-—:]\s*(.+)$", body)
        
        if m2:
            name = m2.group(1).strip()
            desc = m2.group(2).strip()
            print(f"[TEACH] Prepoznata 'Teach: {name} - {desc}' komanda.")
            
            # 2a. Proveri kameru
            if not self.is_camera_on or self.last_frame is None:
                self.append_to_chat("Assistant: Molim Vas, uključite kameru da bih mogao da naučim vizuelno. Kamera mora biti aktivna i prikazivati sliku.")
                return True # Jeste 'Teach' komanda, ali neuspešna
                
            frame_to_crop = self.last_frame.copy()
            crop_bgr = None
            
            try:
                # 2b. Pokušaj detekciju (Prioritet: Lice)
                if hasattr(self, 'face_cascade') and self.face_cascade is not None:
                    gray = cv2.cvtColor(frame_to_crop, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    
                    if len(faces) > 0:
                        (fx, fy, fw, fh) = faces[0] # Uzmi prvo lice
                        crop_bgr = frame_to_crop[fy:fy+fh, fx:fx+fw]
                        print(f"[TEACH] Pronađeno lice za '{name}' komandom iz chata.")

                # 2c. Ako nema lica, pokušaj YOLO
                if crop_bgr is None and self.yolo_model is not None:
                    results = self.yolo_model(frame_to_crop, verbose=False)
                    if results and len(results) > 0 and len(results[0].boxes) > 0:
                        box = results[0].boxes[0] # Uzmi prvu detekciju
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop_bgr = frame_to_crop[y1:y2, x1:x2]
                        print(f"[TEACH] Pronađen YOLO objekat za '{name}' komandom iz chata.")
                
                # 2d. Fallback: Koristi ceo frejm ako ništa nije detektovano
                if crop_bgr is None:
                    crop_bgr = frame_to_crop
                    print(f"[TEACH] Nije detektovano lice ili objekat, koristim ceo frejm za '{name}'.")
                
                # 2e. Generiši embedding i sačuvaj (koristeći novu CLIP funkciju)
                success = self.add_clip_embedding(name, desc, crop_bgr)
                
                if success:
                    self.append_to_chat(f"Assistant: Uspešno naučeno! Sačuvao sam vizuelni uzorak za '{name}'.")
                else:
                    self.append_to_chat(f"Assistant: Pokušao sam da naučim '{name}', ali je došlo do greške prilikom generisanja CLIP embeddinga.")
                
                return True # Komanda je obrađena
            
            except Exception as e:
                print(f"[TEACH] Greška u 'Teach:' komandi: {e}")
                self.append_to_chat(f"Assistant: Došlo je do greške: {e}")
                return True # Jeste bila 'Teach' komanda

        # 3. AKO NIJE NOVI FORMAT, proveri STARI FORMAT (key=value)
        if '=' in body and ( ';' in body or '\n' in body or ',' in body):
            print("[TEACH] Prepoznat stari 'key=value' format.")
            parts = re.split(r'[;,\n]+', body)
            name = None
            desc = None
            context = 'general'
            image_path = None
            
            for part in parts:
                if '=' not in part:
                    continue
                k, v = part.split('=', 1)
                k = k.strip().lower()
                v = v.strip()
                if k == 'name':
                    name = v
                elif k in ('desc', 'description'):
                    desc = v
                elif k == 'context':
                    context = v
                elif k in ('image', 'img', 'image_path'):
                    image_path = v
        
        # 4. Fallback: 'Teach: ime opis' (stari format)
        else:
            print("[TEACH] Prepoznat stari 'fallback' format.")
            name = None
            desc = None
            context = 'general'
            if len(body.split()) == 1:
                name = body.strip()
            else:
                toks = body.split(None, 1)
                if toks:
                    name = toks[0].strip()
                    if len(toks) > 1:
                        desc = toks[1].strip()
            
            if re.search(r'\bvisual\b|\bobject\b|\bimage\b', message, re.IGNORECASE):
                context = 'visual'
                
        if not name:
            return True # Bila je 'Teach' komanda, ali nevalidna

        # --- Logika za STARE formate (tekstualno učenje ili učenje sa putanje) ---
        entry = {
            'description': desc or '',
            'examples': [],
            'context': context,
            'learned_on': datetime.now().isoformat()
        }

        # Ako je data putanja do slike (stara logika), pokušaj da je učitaš
        if image_path:
            image_path = image_path.strip('"')
            if os.path.isabs(image_path):
                img_full = image_path
            else:
                img_full = os.path.join(self.base_path, '..', image_path)
            
            if os.path.exists(img_full):
                entry['visual_data'] = {'image_path': img_full, 'added_on': datetime.now().isoformat()}
                try:
                    img_bgr = cv2.imread(img_full)
                    if img_bgr is not None:
                        # Koristi novu CLIP funkciju
                        added = self.add_clip_embedding(name, desc, img_bgr)
                        entry.setdefault('notes', []).append('Visual sample (CLIP) added via chat import' if added else 'Visual sample (CLIP) failed')
                except Exception as e:
                    print(f"[TEACH] Failed to add visual sample from image: {e}")

        try:
            if getattr(self, 'backend', 'local') == 'ollama':
                lp = f"""Please provide a concise (1-3 sentence) SUMMARY for an object named '{name}'.\nDescription provided: {desc or '(none)'}\nInstructions: Output only the SUMMARY text. Do NOT include chain-of-thought."""
                resp = self.ollama_generate(lp, model=self.ollama_model, max_tokens=200)
                if resp and not resp.startswith('[OLLAMA'):
                    entry['understanding'] = resp.strip()
                # --- FIX: Add fallback if response is empty ---
                if not entry.get('understanding'):
                    entry['understanding'] = f"Learned: {name}. Description: {desc}" if desc else f"Learned: {name}"
                    print(f"[TEACH] Using fallback understanding for {name}")
        except Exception as e:
            print(f"[TEACH] Ollama summary generation failed: {e}")
            entry['understanding'] = f"Learned: {name}. Description: {desc}" if desc else f"Learned: {name}"

        try:
            self.knowledge_base[name] = entry
            self.save_knowledge_base()
            self.append_to_chat(f"Assistant: Learned (text-only) '{name}'. Summary: {entry.get('understanding', entry['description'])}")
            print(f"[TEACH] Saved knowledge for {name} (text or path-based)")
            return True
        except Exception as e:
            print(f"[TEACH] Failed to save knowledge: {e}")
            return True # I dalje je bila 'Teach' komanda
    
    def toggle_camera(self):
        if not self.is_camera_on:
            self.vid = cv2.VideoCapture(0)
            if self.vid.isOpened():
                self.is_camera_on = True
                self.cam_btn.configure(text="Turn Camera Off")
                self.append_to_chat("Assistant: Camera is on! I'll tell you what I see.")
            else:
                self.status.configure(text="Error: Could not open camera")
        else:
            if self.vid:
                self.vid.release()
            self.is_camera_on = False
            self.cam_btn.configure(text="Turn Camera On")
            self.canvas.delete("all")
            self.current_detections = []
            self.append_to_chat("Assistant: Camera is off. You can still chat with me!")
    
    def capture_current_frame(self, context='visual'):
        """Capture the current frame, isolate faces, and show preview for saving."""
        if self.last_frame is None:
            messagebox.showerror("No Frame", "No frame available to capture.")
            return
        frame = self.last_frame.copy()
        faces = []
        # Assume self.current_detections contains detection results with type and bbox
        for det in self.current_detections:
            if det.get('type') == 'face' and 'bbox' in det:
                x, y, w, h = det['bbox']
                face_crop = frame[y:y+h, x:x+w]
                faces.append(face_crop)
        if not faces:
            messagebox.showerror("Face Error", "No face detected in the frame.")
            return
        # Show preview of the first detected face (or let user select if multiple)
        preview_face = faces[0]
        self._update_preview_canvas(preview_face)
        # Save the face crop directly (or after user confirmation)
        # ...existing logic for saving...
        # Optionally, allow saving all detected faces in a loop
        # for idx, face_crop in enumerate(faces):
        #     self._update_preview_canvas(face_crop)
        #     ...save logic...

    def process_and_draw_frame(self, frame):
        """Process frame, draw detections, and filter objects by higher threshold and color logic."""
        frame_copy = frame.copy()
        if self.yolo_model:
            results = self.yolo_model(frame_copy, verbose=False)
            detected = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = r.names[cls]
                    # Strict threshold: only show if conf > 0.7
                    if conf <= 0.7:
                        continue
                    try:
                        crop = frame_copy[y1:y2, x1:x2]
                        final_name = name
                        final_score = conf
                        if crop is not None and crop.size > 0:
                            match_name, match_score = self.match_visual(crop, return_score=True)
                            if match_name:
                                final_name = match_name
                                final_score = match_score
                        if final_name not in detected:
                            detected.append(final_name)
                        # Color: green >0.8, yellow 0.7-0.8, red otherwise
                        if final_score > 0.8:
                            box_color = (0, 255, 0)
                            text_color = (0, 255, 0)
                            line_width = 3
                            text_size = 0.8
                        elif final_score > 0.7:
                            box_color = (0, 255, 255)
                            text_color = (0, 255, 255)
                            line_width = 2
                            text_size = 0.7
                        else:
                            box_color = (0, 0, 255)
                            text_color = (0, 0, 255)
                            line_width = 2
                            text_size = 0.6
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), box_color, line_width)
                        cv2.putText(frame_copy, f"{final_name} {final_score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)
                    except Exception as e:
                        print(f"[ERROR] Detection priority check failed: {e}")
            person_names = []
            try:
                if hasattr(self, 'face_cascade') and self.face_cascade is not None:
                    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    for (fx, fy, fw, fh) in faces:
                        if fw > 0 and fh > 0:
                            face_crop = gray[fy:fy+fh, fx:fx+fw]
                            face_resized = cv2.resize(face_crop, (200, 200))
                            name, conf = self.predict_face(face_resized)
                            if name:
                                person_names.append(name)
                                if name not in detected:
                                    detected.append(name)
                                    if not self.dialog_update_running:
                                        print(f"[FACE] Recognized person: {name} (confidence: {conf:.1f})")
                                # Face rectangle color: green if conf > 90, blue otherwise
                                if conf is not None and conf > 90:
                                    face_color = (0, 255, 0)
                                    text_color = (0, 255, 0)
                                else:
                                    face_color = (255, 0, 0)
                                    text_color = (255, 0, 0)
                                line_width = 3 if conf is not None and conf > 90 else 2
                                text_size = 0.8 if conf is not None and conf > 90 else 0.7
                                cv2.rectangle(frame_copy, (fx, fy), (fx+fw, fy+fh), face_color, line_width)
                                cv2.putText(frame_copy, f"{name} ({conf:.1f})", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)
                                # Greet and load session for new persons
                                if name not in self.greeted_persons:
                                    self.greeted_persons.add(name)
                                    self.append_to_chat(f"Assistant: Hello, {name}! Welcome back.")
                                    # Load personalized session/chat history if available
                                    if hasattr(self, 'chat_history') and self.chat_history:
                                        person_history = [msg for msg in self.chat_history if name in msg]
                                        if person_history:
                                            self.append_to_chat(f"Assistant: Here is your recent session, {name}:\n" + '\n'.join(person_history[-5:]))
                            else:
                                person_names.append("Unknown Person")
                                if "person" in detected and "Unknown Person" not in detected:
                                    detected.append("Unknown Person")
                                cv2.rectangle(frame_copy, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
                                cv2.putText(frame_copy, "Unknown Person", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                                # Greet unknown person
                                if "Unknown Person" not in self.greeted_persons:
                                    self.greeted_persons.add("Unknown Person")
                                    self.append_to_chat("Assistant: Hello, I see a new person! If you'd like to introduce yourself, please type your name.")
            except Exception as e:
                print(f"[FACE] Error in face recognition: {e}")
            try:
                if not detected:
                    match_name = self.match_visual(frame_copy)
                    if match_name:
                        detected.append(match_name)
            except Exception:
                pass
            corrected_detected = []
            for label in detected:
                corrected_label = self.apply_correction(label)
                corrected_detected.append(corrected_label)
            detected = corrected_detected
            if not self.dialog_update_running:
                # Track entry/exit for recognized people
                prev_people = set([d for d in self.current_detections if d in self.knowledge_base and self.knowledge_base[d].get('context') == 'face'])
                new_people = set([d for d in detected if d in self.knowledge_base and self.knowledge_base[d].get('context') == 'face'])
                # Greet only on entry
                for person in new_people - prev_people:
                    if person not in self.greeted_persons:
                        self.greeted_persons.add(person)
                        self.append_to_chat(f"Assistant: Hello, {person}! Welcome back.")
                # Register exit
                for person in prev_people - new_people:
                    if person in self.greeted_persons:
                        self.greeted_persons.remove(person)
                        self.append_to_chat(f"Assistant: {person} has left the camera view.")
                self.current_detections = detected
                print(f"[DETECTION] Updated detections: {detected}")
                if detected:
                    self.status.configure(text=f"Detected: {', '.join(detected)}")
                    recognized_people = [d for d in detected if d in self.knowledge_base and self.knowledge_base[d].get('context') == 'face']
                    objects = [d for d in detected if d not in recognized_people]
                    unknown_people = [d for d in detected if d == "Unknown Person"]
                    description_parts = []
                    if recognized_people:
                        description_parts.append(f"I see {', '.join(recognized_people)}!")
                    if unknown_people:
                        description_parts.append("I also detect an unknown person.")
                    if objects:
                        description_parts.append(f"I also see: {', '.join(objects)}.")
                    description_prompt = " ".join(description_parts) if description_parts else f"Objects detected: {', '.join(detected)}."
                    print(f"[DETECTION] Generating response for: {description_prompt}")
                    if getattr(self, 'auto_describe', False):
                        try:
                            permanent_rules = self.get_permanent_knowledge()
                            context_parts = []
                            if permanent_rules:
                                context_parts.append("Permanent Instructions (Must Follow):")
                                context_parts.extend([f"- {rule}" for rule in permanent_rules])
                                context_parts.append("\n")
                            context_parts.append(f"Scene context: {description_prompt}")
                            context_parts.append("Instruction: Describe this scene in a friendly way.")
                            model_prompt = "\n".join(context_parts)
                            if getattr(self, 'backend', 'local') == 'ollama':
                                response_text = self.ollama_generate(
                                    model_prompt,
                                    model=self.ollama_model,
                                    max_tokens=100
                                )
                            else:
                                response = self.llm(
                                    model_prompt,
                                    max_tokens=100,
                                    stop=["\nUser:", "User:", "<end_of_turn>"]
                                )
                                response_text = response['choices'][0]['text'].strip()
                            if response_text and not response_text.startswith("[OLLAMA"):
                                self.append_to_chat(f"Assistant: {response_text}")
                                print(f"[DETECTION] Response generated: {response_text[:80]}...")
                        except Exception as e:
                            print(f"[DETECTION] Error generating description: {e}")
                    else:
                        print("[DETECTION] auto_describe disabled; skipping LLM call. Press 'Describe Scene' to generate description.")
                else:
                    self.status.configure(text="No objects detected")
        return frame_copy


    def update(self):
        if self.is_camera_on and self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                try:
                    self.last_frame = frame.copy()
                except Exception:
                    self.last_frame = frame
                
                frame_to_process = None
                if self.is_frozen and self.frozen_frame is not None:
                    frame_to_process = self.frozen_frame
                else:
                    frame_to_process = self.last_frame
                
                processed_frame = None
                if frame_to_process is not None:
                    # --- IZMENA: Sačuvaj sirovi frejm pre obrade ---
                    try:
                        self.last_raw_frame_processed = frame_to_process.copy()
                    except Exception:
                        self.last_raw_frame_processed = frame_to_process
                    # --- KRAJ IZMENE ---
                    processed_frame = self.process_and_draw_frame(frame_to_process)
                    self.last_processed_frame = processed_frame 
                
                
                if self.dialog_update_running:
                    # Ne crtaj na glavnom canvasu dok je dialog otvoren
                    pass 
                
                else:
                    # --- IZMENA (Zahtev 1): Logika za skaliranje + PAUZIRANJE KADA JE DIALOG OTVOREN ---
                    if self.dialog_update_running:
                        # Dialog je otvoren - preskoči prikaz na glavnom canvas-u
                        self.canvas.delete("all")
                    elif processed_frame is not None:
                        canvas_w = self.canvas.winfo_width()
                        canvas_h = self.canvas.winfo_height()
                        
                        if canvas_w < 10 or canvas_h < 10:
                             # Canvas još nije vidljiv, sačekaj
                             self.window.after(self.delay, self.update)
                             return

                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_pil = PIL.Image.fromarray(frame_rgb)
                        
                        # .thumbnail() održava aspect ratio
                        frame_pil.thumbnail((canvas_w, canvas_h), PIL.Image.Resampling.LANCZOS)

                        self.photo = PIL.ImageTk.PhotoImage(image=frame_pil)
                        
                        self.canvas.delete("all")
                        # Centrira sliku unutar canvasa
                        self.canvas.create_image(
                            canvas_w / 2, 
                            canvas_h / 2, 
                            image=self.photo, 
                            anchor=tk.CENTER
                        )
        
        elif not self.is_camera_on:
            try:
                self.canvas.delete("all")
            except Exception:
                pass
        
        self.window.after(self.delay, self.update)

    # --- IZMENA (Zahtev 2): Uklonjena 'apply_orb_settings' ---

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = DetectionGUI(root, "Smart Vision Assistant")
        app.load_models()
        app.delay = 15
        if hasattr(app, 'chat_display'):
            # --- ISPRAVKA (Log greška): Popravljen prelomljen string ---
            app.append_to_chat("Assistant: Hello! I'm your worker. You can chat with me anytime, "
                               "and when you turn on the camera, I'll also tell you what I see.")
        
        # --- IZMENA (Zahtev 1): Uklonjen 'on_main_configure' i .bind() ---
        # Logika skaliranja je sada u potpunosti unutar self.update()
        # zahvaljujući 'fill=BOTH, expand=True' na canvasu.
        
        app.update()
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {e}")
        # import traceback # <-- Ovo je sada na vrhu fajla
        print(f"TRACEBACK: {traceback.format_exc()}")
        if 'root' in locals():
            root.destroy()