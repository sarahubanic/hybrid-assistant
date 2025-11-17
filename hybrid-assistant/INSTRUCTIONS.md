# Instructions — Setup & Run

This document lists exact prerequisites, environment setup, and commands to run the hybrid assistant on Windows (PowerShell). Follow these steps to get a working local environment.

Prerequisites
- Windows (tested with PowerShell 5.1).
- Python 3.10 or 3.11 (virtualenv recommended).
- Ollama Desktop / Ollama server (recommended for local LLM + embeddings).
  - Recommended models to install in Ollama: `gemma3:1b` (chat) and `nomic-embed-text:latest` (embeddings).
- A webcam for the GUI.

Recommended Python packages
- The project includes `requirements.txt`. It typically contains:
  - opencv-contrib-python (for face LBPH)
  - pillow
  - numpy
  - requests
  - ultralytics (YOLOv8)
  - torch (CPU)
  - transformers (optional for CLIP)

Quick setup (PowerShell)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Install or ensure Ollama is running (if using Ollama backend):

- If using Ollama Desktop, launch the app so the HTTP API is available.
- Pull recommended models with the Ollama CLI (example):

```powershell
ollama pull gemma3:1b
ollama pull nomic-embed-text:latest
```

Notes about Ollama
- Default Ollama HTTP endpoints used by the app:
  - Generation: `http://127.0.0.1:11434/api/generate`
  - Embeddings: `http://127.0.0.1:11434/api/embed`
- If Ollama is not running, the assistant will still function for basic GUI/camera operations, but chat and semantic retrieval features will be limited.

Environment variables (optional)
- `HA_BACKEND` — `'ollama'` (default) or `'local'`.
- `HA_CHAT_MODEL` — model name for chat (e.g., `gemma3:1b`).
- `HA_EMBED_MODEL` — embedding model name (e.g., `nomic-embed-text:latest`).
- `OLLAMA_URL` / `OLLAMA_EMBED_URL` — override default Ollama endpoints.

Running the GUI

From the repo root (PowerShell, with venv activated):

```powershell
.\.venv\Scripts\Activate.ps1
python src/detection_gui.py
```

Or use the provided VS Code tasks (if configured) or the batch scripts in the root (Windows):

```powershell
start_assistant.bat
# or
start_visual.bat
```

Teaching workflows
- Visual: Click `Teach Me Something` in the GUI, freeze the frame, draw a rectangle or polygon on the dialog image. Save the area and provide a name/description.
- Chat: Use `Teach: name - description` in the chat input to add a quick knowledge entry.

Troubleshooting
- If face recognition fails, ensure `opencv-contrib-python` is installed (this provides `cv2.face`).
- If the camera does not open, try to run another camera test app or check camera permissions.
- If Ollama returns connection errors, verify Ollama Desktop or server is running and reachable at `127.0.0.1:11434`.

Advanced notes
- For better visual matching, use CLIP embeddings (the GUI supports CLIP if `transformers` and `torch` are installed). The default `requirements.txt` may not include GPU-enabled torch; install appropriate `torch` build for your hardware if needed.

If you want, I can:
- Add a short troubleshooting script to verify Ollama connectivity and model availability.
- Create a single-step `run_assistant.ps1` that activates the venv and starts the GUI and (optionally) Ollama in the background.

---
Updated: November 17, 2025
