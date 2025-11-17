"""
Realtime assistant server (FastAPI) - starter

Features:
- Serves a small HTML UI (`/`) with Start/Stop buttons.
- WebSocket endpoint `/ws` that streams structured JSON interpretations to connected clients.
- REST endpoints `/start` and `/stop` to control autonomous detection.
- Best-effort GGUF loader: tries to import `gguf` or to load via `transformers`. If LLM isn't available, server still streams structured JSON detections.

Notes:
- This is a starter/harness. Adapt to your project's auth and runtime.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import threading

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Try to import gguf (pygguf/gguf-py) to read metadata (best-effort)
try:
    import gguf
    GGUF_PY_AVAILABLE = True
except Exception:
    GGUF_PY_AVAILABLE = False

app = FastAPI()

# serve static UI
static_dir = Path(__file__).resolve().parent.parent / 'static'
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast_json(self, message: dict):
        payload = json.dumps(message)
        for conn in list(self.active):
            try:
                await conn.send_text(payload)
            except Exception:
                self.disconnect(conn)

manager = ConnectionManager()

# Global control for detection loop
_detection_task: Optional[asyncio.Task] = None
_detection_thread_stop = threading.Event()
_detection_lock = threading.Lock()

# Model placeholders
yolo_model = None
llm_model = None
llm_tokenizer = None

GGUF_MODEL_PATH = os.environ.get('GGUF_MODEL_PATH', 'models/model.gguf')
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', 'yolov8n.pt')


def try_load_yolo(weights: str = YOLO_WEIGHTS):
    global yolo_model
    if YOLO is None:
        print('Ultralytics not available; detection disabled.')
        return None
    try:
        yolo_model = YOLO(weights)
        return yolo_model
    except Exception as e:
        print('Failed to load YOLO weights:', e)
        return None


def load_llm_best_effort(gguf_path: str = GGUF_MODEL_PATH):
    """Best-effort LLM loader. Try gguf (pygguf) for metadata, then transformers for inference.
    Returns (model, tokenizer) or (None, None).
    """
    global llm_model, llm_tokenizer
    if TRANSFORMERS_AVAILABLE:
        try:
            print('Trying to load via transformers from', gguf_path)
            tokenizer = AutoTokenizer.from_pretrained(gguf_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(gguf_path, device_map='auto')
            llm_model, llm_tokenizer = model, tokenizer
            print('Loaded LLM via transformers')
            return model, tokenizer
        except Exception as e:
            print('Transformers gguf load failed:', e)
    # Try to extract a model name from the file path as a reliable fallback
    try:
        if Path(gguf_path).exists():
            file_stem = Path(gguf_path).stem
            print(f"Detected GGUF model file: {Path(gguf_path).name} -> model id '{file_stem}'")
            # If gguf-py is available, attempt to read metadata name fields (best-effort)
            if GGUF_PY_AVAILABLE:
                try:
                    reader = gguf.GGUFReader(gguf_path)
                    # Attempt to find metadata fields containing 'name'
                    name_keys = [k for k in reader.fields.keys() if 'name' in k.lower()]
                    meta_name = None
                    for k in name_keys:
                        try:
                            field = reader.fields[k]
                            # try to read a textual representation from the field parts
                            part = None
                            if hasattr(field, 'parts') and len(field.parts) > 0:
                                part = field.parts[0]
                            if isinstance(part, (bytes, bytearray)):
                                meta_name = part.decode('utf-8', 'ignore')
                            else:
                                meta_name = str(part)
                            if meta_name:
                                print(f"GGUF metadata model name (from '{k}'): {meta_name}")
                                break
                        except Exception:
                            continue
                except Exception as e:
                    print('Failed to read GGUF with gguf-py:', e)
        else:
            print(f"GGUF path does not exist: {gguf_path}")
    except Exception:
        pass
    print('No LLM available in this environment. Interpretations will be basic structured JSON only.')
    return None, None


def format_detection_result(result, frame_index: int):
    # Minimal consistent formatting: mirror starter runner schema
    objects = []
    try:
        boxes = getattr(result, 'boxes', None)
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy.tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy)
                confv = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
                cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
                name = yolo_model.names.get(cls, str(cls)) if yolo_model and hasattr(yolo_model, 'names') else str(cls)
                objects.append({
                    'label': name,
                    'confidence': round(confv, 3),
                    'bbox': [round(x, 1) for x in xyxy]
                })
    except Exception:
        # fallback
        pass

    return {
        'frame_index': frame_index,
        'timestamp': f'frame_{frame_index}',
        'objects': objects
    }


def detection_loop(source='0', conf=0.25):
    """Runs in a background thread. Streams detection results to connected WebSocket clients."""
    global _detection_thread_stop, yolo_model
    print('Detection loop started, source=', source)
    if yolo_model is None:
        try_load_yolo()
    if yolo_model is None:
        print('No YOLO model loaded; exiting detection loop')
        return

    frame_idx = 0
    try:
        # use ultralytics streaming API if available
        for res in yolo_model.predict(source=source, conf=conf, imgsz=640, stream=True):
            if _detection_thread_stop.is_set():
                break
            formatted = format_detection_result(res, frame_idx)
            structured = {
                'video': str(source),
                'time_range': None,
                'interpretations': [
                    {
                        'timestamp': formatted['timestamp'],
                        'objects': formatted['objects'],
                        'summary': (f"{formatted['objects'][0]['label']} with confidence {formatted['objects'][0]['confidence']}") if formatted['objects'] else 'No detections'
                    }
                ],
                'global_summary': 'Autonomous stream'
            }
            # send to websocket clients (schedule on event loop)
            asyncio.run_coroutine_threadsafe(manager.broadcast_json(structured), asyncio.get_event_loop())
            frame_idx += 1
    except Exception as e:
        print('Detection loop error:', e)
    finally:
        print('Detection loop stopped')


@app.get('/', response_class=HTMLResponse)
async def index():
    html_file = Path(__file__).resolve().parent.parent / 'static' / 'realtime_ui.html'
    if html_file.exists():
        return FileResponse(str(html_file))
    return HTMLResponse('<html><body><h3>UI not found. Create static/realtime_ui.html</h3></body></html>')


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # keep the connection alive; clients may send control messages in future
            data = await websocket.receive_text()
            # echo or ignore
            # Optionally accept ping/pong
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post('/start')
async def start_detection(request: Request):
    global _detection_task, _detection_thread_stop
    body = await request.json() if request.headers.get('content-type', '').startswith('application/json') else {}
    source = body.get('source', 0)  # default to camera

    with _detection_lock:
        if _detection_task is not None and not _detection_task.done():
            return {'status': 'already_running'}
        _detection_thread_stop.clear()
        loop = asyncio.get_event_loop()
        # run detection loop in separate thread to use Ultrayltics stream
        t = threading.Thread(target=detection_loop, kwargs={'source': source}, daemon=True)
        t.start()
        _detection_task = None
        return {'status': 'started'}


@app.post('/stop')
async def stop_detection():
    global _detection_task, _detection_thread_stop
    _detection_thread_stop.set()
    return {'status': 'stopping'}


@app.post('/load_llm')
async def load_llm_endpoint():
    model, tokenizer = load_llm_best_effort(GGUF_MODEL_PATH)
    return {'llm_loaded': (model is not None and tokenizer is not None), 'gguf_available': GGUF_PY_AVAILABLE}


if __name__ == '__main__':
    import uvicorn
    print('Run server: uvicorn tools.realtime_server:app --reload --port 8000')
    uvicorn.run('tools.realtime_server:app', host='0.0.0.0', port=8000, reload=True)
