# Setup & Configuration Guide

This guide walks you through installing, configuring, and running the Hybrid Assistant.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Ollama Setup](#ollama-setup)
4. [Face Recognition Setup](#face-recognition-setup)
5. [Running the App](#running-the-app)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **OS**: Windows 10+, macOS, or Linux
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU** (optional): NVIDIA GPU with CUDA support for faster inference
- **Camera**: Webcam or USB camera (optional, for visual learning)
- **Disk Space**: 3-8GB (for models and cache)

### Software Requirements

- Python 3.11+
- pip (Python package manager)
- Git (to clone the repo)
- Ollama (for LLM backend)

### Check Your Setup

```bash
# Check Python version
python --version  # Should be 3.11+

# Check pip
pip --version

# Check Git
git --version
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sarahubanic/hybrid-assistant.git
cd hybrid-assistant
```

### Step 2: Run the Launcher

**Windows:**
```bash
# Just double-click run.bat
# OR use command line:
.\run.bat
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python detection_gui.py
```

### Step 3: Answer Setup Questions

The launcher will:
1. Create a virtual environment (if needed)
2. Install all dependencies
3. Run pre-flight checks
4. Ask you to select a mode (CPU/CUDA/Hybrid)

## Ollama Setup

Ollama provides the local LLM backend. The app works **completely offline** after initial setup.

### Install Ollama

1. **Download** from https://ollama.ai
2. **Install** for your OS
3. **Verify** installation:
   ```bash
   ollama --version
   ```

### Start Ollama

**Windows:**
```bash
# Ollama runs as a service automatically after install
# Or start manually:
ollama serve
```

**Linux/macOS:**
```bash
ollama serve
```

Ollama will be available at: `http://localhost:11434`

### Download Models

The app works best with smaller models on local hardware. Download these recommended models:

```bash
# Recommended for chat (small & fast)
ollama pull gemma2:2b

# Alternative (better quality, slower)
ollama pull mistral:latest

# For embeddings (RECOMMENDED - most important)
ollama pull nomic-embed-text:latest

# For advanced users (larger, slower)
ollama pull llama2:13b
ollama pull neural-chat:latest
```

**Recommended Setup for Most Users:**
```bash
ollama pull gemma2:2b
ollama pull nomic-embed-text:latest
```

**Storage Requirements:**
- `gemma2:2b`: 1.4GB
- `mistral:latest`: 4GB
- `llama2:13b`: 7GB
- `nomic-embed-text`: 200MB

### Update Configuration

Edit `~/.continue/config.yaml` to set your preferred model:

```yaml
models:
  - name: Gemma2 2B
    provider: ollama
    model: gemma2:2b
    apiBase: http://localhost:11434
    
  - name: Mistral Chat
    provider: ollama
    model: mistral:latest
    apiBase: http://localhost:11434

embeddingsProvider:
  provider: ollama
  model: nomic-embed-text:latest
  apiBase: http://localhost:11434
```

## Face Recognition Setup

Face recognition is **optional** but requires the opencv-contrib-python package.

### Install Face Recognition

The `opencv-contrib-python` package is in `requirements.txt` and installs automatically.

To verify:
```bash
python -c "import cv2; print(cv2.__version__)"
python -c "import cv2.face; print('Face module available')"
```

### How Face Recognition Works

1. **First time you see a face**: Trained from camera using LBPH (Local Binary Patterns Histograms)
2. **Teach via GUI**: Draw a rectangle around the face and give it a name
3. **Persistent**: Face models saved in `learned_items/faces/` as `.yml` files
4. **Privacy**: All local, no cloud processing

### Train Face Recognition

In the app:
1. Click **"Teach"** button
2. Draw rectangle around a face
3. Enter the person's name
4. Click **Save**
5. Repeat with different angles/lighting for better accuracy

**Tips for better face recognition:**
- Train with multiple photos (front, side angles)
- Use different lighting conditions
- Include several samples per person (3-5 photos minimum)
- Keep faces clear and visible in frame

## Running the App

### Quick Start (Windows)

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run the app
.\run.bat
```

Then:
1. Select mode: **1** (CPU), **2** (CUDA), or **3** (Hybrid)
2. Wait for app to load
3. Click **"Teach"** to start teaching objects/faces

### Command Line (Linux/macOS)

```bash
# Terminal 1
ollama serve

# Terminal 2
source venv/bin/activate
python detection_gui.py
```

### What Each Mode Does

| Mode | Use Case | Speed | Requirements |
|------|----------|-------|--------------|
| **CPU** | Safe, universal | Slower | None |
| **CUDA** | Fast on NVIDIA GPU | Fast | NVIDIA GPU + CUDA |
| **Hybrid** | Auto-detect | Fast if GPU available | Optional GPU |

## Configuration

### Camera Settings

In `detection_gui.py`, modify these settings:

```python
CAMERA_ID = 0  # Change if using multiple cameras
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
FPS = 30
```

### Confidence Thresholds

```python
FACE_CONFIDENCE_THRESHOLD = 0.5
DETECTION_CONFIDENCE_THRESHOLD = 0.5
```

### Model Paths

Models are cached in:
- **CLIP**: `~/.cache/hybrid-assistant/huggingface/`
- **Ollama**: `~/.ollama/models/`
- **App Data**: `./learned_items/`

### Offline Mode

Force offline mode (uses cached models only):

```bash
set TRANSFORMERS_OFFLINE=1
python detection_gui.py
```

## Troubleshooting

### "Camera not found"

**Solution:**
```bash
# Check available cameras
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAILED')"

# Try different camera ID (0, 1, 2...)
# Edit detection_gui.py: CAMERA_ID = 1
```

### "Ollama connection failed"

**Solution:**
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# If failed, start Ollama
ollama serve

# Check port is not in use
netstat -an | grep 11434
```

### "CLIP model download failed"

**Solution:**
- First run needs internet (~500MB download)
- After first run, works completely offline
- If stuck: Delete cache and retry
  ```bash
  rm -r ~/.cache/hybrid-assistant/
  python detection_gui.py
  ```

### "Out of memory / App crashes"

**Solutions:**
1. Use smaller model: `gemma2:2b` instead of `llama2:13b`
2. Use CPU mode instead of CUDA
3. Close other applications
4. Increase virtual memory / swap

**Check memory usage:**
```bash
# Windows
tasklist /FI "IMAGENAME eq python.exe"

# Linux/macOS
ps aux | grep python
```

### "Face recognition not working"

**Solutions:**
1. Ensure lighting is good
2. Train with multiple angles/positions
3. Check that `opencv-contrib-python` is installed:
   ```bash
   pip install opencv-contrib-python
   ```
4. Verify face models were saved in `learned_items/faces/`

### "Slow performance / Low FPS"

**Solutions:**
1. Use Hybrid or CUDA mode (not CPU)
2. Use smaller models
3. Reduce display resolution
4. Close other GPU-intensive applications
5. Check GPU is being used:
   ```bash
   # NVIDIA GPU
   nvidia-smi
   ```

### "App won't start"

**Steps to debug:**
1. Run pre-flight check:
   ```bash
   python startup_check.py
   ```
2. Check logs in `logs/` folder
3. Verify Python version: `python --version`
4. Reinstall requirements:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### "Models not downloading"

**Solution:**
- First run requires internet (~500-600MB)
- Check internet connection
- Try manual download:
  ```bash
  python -c "import clip; clip.load('ViT-B/32')"
  ollama pull nomic-embed-text:latest
  ```

## Advanced Configuration

### Custom Ollama Server

To use a remote Ollama server:

1. Edit `~/.continue/config.yaml`:
   ```yaml
   apiBase: http://192.168.1.100:11434
   ```

2. Or set environment variable:
   ```bash
   set OLLAMA_HOST=http://remote-server:11434
   ```

### Performance Optimization

**For low-end hardware:**
```yaml
# Use smallest models
model: gemma2:2b
contextWindow: 2048
maxTokens: 512
```

**For high-end hardware:**
```yaml
# Use larger models
model: llama2:13b
contextWindow: 8192
maxTokens: 2048
```

### Batch Processing

Process multiple images for visual learning:

```bash
python tools/batch_add_visual.py --folder path/to/images --label object_name
```

## Need Help?

1. **Check logs**: `logs/assistant_*.log`
2. **Run diagnostics**: `python startup_check.py`
3. **Offline mode**: See [OFFLINE_MODE.md](OFFLINE_MODE.md)
4. **Issues on GitHub**: https://github.com/sarahubanic/hybrid-assistant/issues

---

**Last Updated**: November 17, 2025

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
