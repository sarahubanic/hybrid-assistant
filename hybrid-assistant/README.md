# Hybrid Assistant

This repository contains a local, privacy-friendly hybrid assistant that combines a vision stack (camera, YOLO, OpenCV face recognition, ORB-based visual matching) with a local LLM backend (Ollama) to provide scene descriptions, object recognition, and an interactive GUI for teaching new objects.

**Current state (Nov 2025)**
- A Tkinter GUI (`src/detection_gui.py`) that shows live camera feed, detection overlays, and a chat panel.
- Local LLM backend integrations via Ollama HTTP API for text generation and embeddings.
- Teach workflows:
  - `Add Object` GUI: draw a rectangle on the live camera to save a visual sample and optional description.
  - `Teach:` chat parser: type `Teach: name - description` (or key=value pairs) to save knowledge from chat.
- Visual DB persisted under `learned_items/visual_db.pkl` and image samples in `learned_items/visual_samples/<label>/`.
- Batch visual import utility: `tools/batch_add_visual.py` (collect many labeled images into the visual DB).

**Important**: The GUI no longer shows ORB tuning controls; teaching is done by drawing rectangles or using the `Teach:` chat shorthand.

**Table of Contents**
- **Overview**
- **Requirements**
- **Quick Setup**
- **Running the assistant**
- **GUI: Key features & how to use**
- **Teach from chat**
- **Add Object (rectangle)**
# Hybrid Assistant

This repository provides a local, privacy-focused hybrid assistant combining a camera-driven visual UI and a chat backend.

Key features
- Live camera view with detection overlays and a chat panel (`src/detection_gui.py`).
- Teach the assistant visually (rectangle/polygon) or via chat (`Teach:` commands).
- Visual learning via CLIP embeddings and optional ORB fallback.
- Face recognition using OpenCV LBPH when `opencv-contrib-python` is installed.
- Ollama HTTP API integration for text generation and embeddings (recommended).

Files of primary interest
- `src/detection_gui.py` — main GUI application (camera, chat, teach workflows).
- `learned_items/knowledge.json` — persistent knowledge base.
- `learned_items/visual_samples/` — saved visual crops.
- `tools/` — helper scripts (batch import, visual DB tools).

See `INSTRUCTIONS.md` for step-by-step setup, prerequisites (including Ollama and recommended models), and run commands.

Last updated: November 17, 2025
  - Add desired models to Ollama (e.g., `gemma3:1b`, `mistral:latest`, `nomic-embed-text:latest`).
