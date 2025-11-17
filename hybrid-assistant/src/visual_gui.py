import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
from datetime import datetime
import pickle
from ultralytics import YOLO
from detection_gui import DetectionGUI  # Reuse existing detection functionality

class VisualRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Recognition System")
        
        # Initialize detection system
        self.detection = DetectionGUI(self.root, "Visual Recognition System")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.root.quit()
            return
        
        # Set up video source for detection GUI
        self.detection.vid = self.cap
        self.detection.is_camera_on = True
        
        # Initialize face recognition
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.detection.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.detection.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detection.face_labels = {}
        self.detection.next_label = 0
        
        # Load any existing face recognition data
        face_data_path = os.path.join(self.detection.learning_dir, "face_recognizer.yml")
        face_labels_path = os.path.join(self.detection.learning_dir, "face_labels.pkl")
        
        if os.path.exists(face_data_path) and os.path.exists(face_labels_path):
            self.detection.face_recognizer.read(face_data_path)
            with open(face_labels_path, 'rb') as f:
                self.detection.face_labels = pickle.load(f)
                if self.detection.face_labels:
                    self.detection.next_label = max(self.detection.face_labels.values()) + 1
        
        # Initialize YOLO
        yolo_path = os.path.join(self.detection.models_dir, "yolov8n.pt")
        if os.path.exists(yolo_path):
            self.detection.yolo_model = YOLO(yolo_path)
        
        # Initialize ORB detector for visual recognition
        self.detection.orb = cv2.ORB_create()
        self.detection.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.detection.visual_samples = {}
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.root.quit()
            return
            
        # Create GUI elements
        self.create_gui()
        
        # Initialize flags and data
        self.running = True
        self.learning_mode = False
        self.current_name = ""
        self.recognition_enabled = True
        
        # Start update loop
        self.update_frame()

    def create_gui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Recognition controls
        recognition_frame = ttk.LabelFrame(left_panel, text="Recognition Controls")
        recognition_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.toggle_btn = ttk.Button(recognition_frame, text="Toggle Recognition",
                                   command=self.toggle_recognition)
        self.toggle_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Learning controls
        learning_frame = ttk.LabelFrame(left_panel, text="Learning Controls")
        learning_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(learning_frame, text="Name:").pack(fill=tk.X, padx=5, pady=2)
        self.name_entry = ttk.Entry(learning_frame)
        self.name_entry.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.learn_btn = ttk.Button(learning_frame, text="Learn Face/Object",
                                  command=self.start_learning)
        self.learn_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Learning mode frame
        self.learning_frame = ttk.LabelFrame(left_panel, text="Learning Mode")
        
        self.face_btn = ttk.Button(self.learning_frame, text="Capture Face",
                                 command=lambda: self.capture_sample('face'))
        self.face_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.visual_btn = ttk.Button(self.learning_frame, text="Capture Visual",
                                   command=lambda: self.capture_sample('visual'))
        self.visual_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.cancel_btn = ttk.Button(self.learning_frame, text="Cancel",
                                   command=self.cancel_learning)
        self.cancel_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Status frame
        status_frame = ttk.Frame(left_panel)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready")
        self.status_label.pack(pady=5)
        
        # Right panel - Video feed
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(right_panel)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # Process frame with detection
                if self.recognition_enabled:
                    # Run face recognition
                    faces = self.detection.detect_faces(frame)
                    if faces is not None:
                        for (x, y, w, h) in faces:
                            face_roi = frame[y:y+h, x:x+w]
                            # Try to recognize face
                            name = self.detection.recognize_face(face_roi)
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, name, (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Try visual recognition if no YOLO detections
                    results = self.detection.yolo_model(frame)
                    if len(results[0].boxes) == 0:
                        visual_match = self.detection.match_visual(frame)
                        if visual_match:
                            name, score, box = visual_match
                            x, y, w, h = box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(frame, f"{name} ({score:.2f})", (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
                self.video_label.configure(image=img)
                self.video_label.image = img
            
            self.root.after(10, self.update_frame)

    def toggle_recognition(self):
        self.recognition_enabled = not self.recognition_enabled
        status = "enabled" if self.recognition_enabled else "disabled"
        self.status_label.config(text=f"Status: Recognition {status}")

    def start_learning(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
            
        self.current_name = name
        self.learning_mode = True
        self.learning_frame.pack(fill=tk.X, pady=(0, 10))
        self.learn_btn.state(['disabled'])
        self.status_label.config(text="Status: Learning Mode - Select capture type")

    def capture_sample(self, context):
        ret, frame = self.cap.read()
        if ret:
            knowledge_entry = {
                "name": self.current_name,
                "type": "visual_data",
                "timestamp": datetime.now().isoformat(),
                "visual_data": True,
                "notes": []
            }
            
            if context == 'face':
                self.detection.process_face_sample(frame, self.current_name, knowledge_entry)
            else:  # visual
                self.detection.process_visual_sample(frame, self.current_name, knowledge_entry)
                
            if knowledge_entry.get("notes"):
                messagebox.showinfo("Success", "\n".join(knowledge_entry["notes"]))
            else:
                messagebox.showwarning("Warning", "Failed to process sample")

    def cancel_learning(self):
        self.learning_mode = False
        self.current_name = ""
        self.learning_frame.pack_forget()
        self.learn_btn.state(['!disabled'])
        self.status_label.config(text="Status: Ready")

    def on_closing(self):
        self.running = False
        if hasattr(self, 'detection'):
            if hasattr(self.detection, 'cap'):
                self.detection.cap.release()
        self.root.quit()

def main():
    try:
        root = tk.Tk()
        root.geometry("1200x800")  # Set a reasonable initial window size
        app = VisualRecognitionGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    main()