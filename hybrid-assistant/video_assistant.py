"""
Simple starter script: run video recognition (Ultralytics YOLO) and format detections
for interpretation by a GGUF-based assistant model.

Behavior:
- Detect objects in a video with Ultralytics YOLO.
- Produce a structured JSON containing timestamps and detections.
- Optionally (if transformers + gguf support present) send a prompt + JSON to the local LLM for interpretation.

Notes:
- The script is intentionally conservative: if the local environment does not support loading the GGUF model into Transformers, it will still run detections and print structured output for later consumption by an assistant process.
- Configure your GGUF model path via `GGUF_MODEL_PATH` environment variable if needed. The script assumes the file name hasn't changed.

This is a starter/harness — adapt to your project's runtime and model-loading preferences.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

# Optional imports for LLM -- best-effort. If unavailable, we fallback.
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

DEFAULT_GGUF_PATH = os.environ.get("GGUF_MODEL_PATH", "models/model.gguf")
DEFAULT_YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")


def run_yolo_on_video(video_path: str, yolo_weights: str = DEFAULT_YOLO_WEIGHTS, conf: float = 0.25):
    """Run Ultralytics YOLO on the provided video and return structured detections.

    Returns:
      dict with keys: video, frames(list of {frame_index, timestamp, objects})
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics package is not available in this environment. Activate the project's venv or install ultralytics.")

    model = YOLO(yolo_weights)

    results = []
    # Use streaming inference to iterate frames; Ultrayltics returns Results objects
    for i, r in enumerate(model.predict(source=video_path, conf=conf, imgsz=640, stream=True)):
        timestamp = None
        try:
            # r.orig_img has shape, r.orig_shape etc. Some Results objects include 'orig_img' and 'time' info
            # We set timestamp as frame index (integrators can map to real time separately)
            timestamp = f"frame_{i}"
            objects = []
            boxes = getattr(r, 'boxes', None)
            if boxes is not None:
                for box in boxes:
                    # boxes in ultralytics v8 expose .xyxy, .conf, .cls
                    xyxy = box.xyxy.tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy)
                    confv = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
                    name = model.names.get(cls, str(cls)) if hasattr(model, 'names') else str(cls)
                    objects.append({
                        "label": name,
                        "confidence": round(confv, 3),
                        "bbox": [round(x, 1) for x in xyxy]
                    })
            else:
                # fallback: use r.boxes.xyxy if attribute differs
                try:
                    for xyxy, confv, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                        objects.append({
                            "label": model.names.get(int(cls), str(int(cls))) if hasattr(model, 'names') else str(int(cls)),
                            "confidence": round(float(confv), 3),
                            "bbox": [round(float(x), 1) for x in xyxy]
                        })
                except Exception:
                    objects = []

            results.append({
                "frame_index": i,
                "timestamp": timestamp,
                "objects": objects
            })
        except Exception as e:
            # keep going on single-frame parse errors
            results.append({"frame_index": i, "timestamp": f"frame_{i}", "objects": [], "error": str(e)})

    return {"video": Path(video_path).name, "frames": results}


def format_for_interpretation(detections: dict, start: float = None, end: float = None):
    """Format detections into the JSON schema expected by the assistant. Returns JSON string."""
    time_range = None
    if start is not None or end is not None:
        time_range = f"{start if start is not None else '0'}-{end if end is not None else 'end'}"

    interpretations = []
    for f in detections.get('frames', []):
        objects = f.get('objects', [])
        if len(objects) == 0:
            continue
        # Simple per-frame summary: top label by confidence
        top = max(objects, key=lambda o: o['confidence'])
        summary = f"{top['label']} with confidence {top['confidence']}"
        interpretations.append({
            "timestamp": f.get('timestamp', f.get('frame_index')),
            "objects": objects,
            "summary": summary
        })

    global_summary = "; ".join([i['summary'] for i in interpretations[:3]]) if interpretations else "No detections"

    out = {
        "video": detections.get('video'),
        "time_range": time_range,
        "interpretations": interpretations,
        "global_summary": global_summary
    }
    return out


def load_llm_if_available(gguf_path: str):
    """Try to load the GGUF model in transformers. If not possible, return None.

    This function is best-effort; many environments won't support direct GGUF loading into HF.
    Integrators can replace this with the project's preferred loader.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available; skipping LLm load.")
        return None, None

    # Try to load via AutoTokenizer + AutoModelForCausalLM from a local folder or identifier.
    try:
        print(f"Attempting to load LLM from: {gguf_path}")
        # If gguf support exists in transformers in this env, the call below may succeed
        tokenizer = AutoTokenizer.from_pretrained(gguf_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(gguf_path, device_map='auto')
        print("LLM loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print("Could not load LLM via transformers from gguf path:", e)
        return None, None


def interpret_with_llm(model, tokenizer, structured_json: dict, prompt_system_file: str = None):
    """Send structured JSON to the LLM for interpretation using a short wrapper prompt.
    Returns the model's raw string output.
    If model is None, return the JSON as a pretty-printed string.
    """
    if model is None or tokenizer is None:
        return json.dumps(structured_json, indent=2)

    # Build messages in chat format (required for llama.cpp + chat templates like Gemma)
    messages = []
    
    # Add system prompt if provided
    if prompt_system_file and Path(prompt_system_file).is_file():
        system_prompt = Path(prompt_system_file).read_text().strip()
        messages.append({"role": "system", "content": system_prompt})
    
    # Add user message with the detection JSON
    user_content = f"Interpret the following video detections. Return only valid JSON following the schema exactly:\n\n{json.dumps(structured_json, ensure_ascii=False, indent=2)}"
    messages.append({"role": "user", "content": user_content})
    
    # DEBUG: print what we're sending to the LLM
    print("DEBUG >>> LLM INPUT MESSAGES:", json.dumps(messages, ensure_ascii=False), flush=True)

    # Use a simple generate pipeline for text generation
    try:
        gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if hasattr(model, 'device') else -1)
        # Pass messages as a list for chat-aware models
        out = gen(messages, max_new_tokens=512, do_sample=False)
        text = out[0]['generated_text'] if isinstance(out[0], dict) and 'generated_text' in out[0] else str(out)
        
        # If output is a list of messages (chat format), extract the assistant message
        if isinstance(text, list):
            for msg in text:
                if msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
        
        print("DEBUG >>> LLM OUTPUT (raw):", repr(text), flush=True)
        
        # Postprocess: extract JSON substring if possible (best-effort)
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1:
            json_out = text[first_brace:last_brace+1]
            print("DEBUG >>> Extracted JSON:", json_out, flush=True)
            return json_out
        
        # If no JSON found but text exists, return it
        if text.strip():
            return text
        
        print("DEBUG >>> LLM returned empty output; returning fallback JSON.", flush=True)
        return json.dumps(structured_json, indent=2)
    except Exception as e:
        print("DEBUG >>> LLM call exception:", repr(e), flush=True)
        return f"LLM call failed: {e}\nFallback JSON:\n" + json.dumps(structured_json, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run video detection and interpret with (optional) GGUF LLM')
    parser.add_argument('video', help='Path to video file to analyze')
    parser.add_argument('--gguf', help='Path to GGUF LLM model (folder or file)', default=DEFAULT_GGUF_PATH)
    parser.add_argument('--yolo', help='YOLO weights to use for detection', default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument('--prompt', help='Path to system prompt file (e.g. starter_prompts/video_assistant_prompt.md)', default='starter_prompts/video_assistant_prompt.md')
    parser.add_argument('--no-llm', help='Do not attempt to load or call the LLM', action='store_true')
    args = parser.parse_args()

    if not Path(args.video).is_file():
        print('Video not found:', args.video)
        sys.exit(2)

    # Step 1: run detection
    print('Running object detection on video...')
    try:
        detections = run_yolo_on_video(args.video, yolo_weights=args.yolo)
    except Exception as e:
        print('Detection failed:', e)
        sys.exit(3)

    structured = format_for_interpretation(detections)

    # Step 2: attempt to load LLM (best-effort)
    model = tokenizer = None
    if not args.no_llm:
        model, tokenizer = load_llm_if_available(args.gguf)

    # Step 3: interpret
    print('\n----- INTERPRETATION OUTPUT -----\n')
    out = interpret_with_llm(model, tokenizer, structured, prompt_system_file=args.prompt)
    print(out)


if __name__ == '__main__':
    main()
