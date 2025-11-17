# Offline Mode Guide

Your Hybrid Assistant can run **completely offline** after initial setup.

## Quick Summary

- **First run**: Requires internet (~500MB download)
- **After first run**: 100% offline capable
- **Required for offline**: Ollama running locally + cached models
- **Cache location**: `C:\Users\<YourUser>\.cache\hybrid-assistant\`

## First Run (With Internet)

```powershell
# Clone and navigate
git clone https://github.com/sarahubanic/hybrid-assistant.git
cd hybrid-assistant

# Just double-click run.bat
# OR use command line
.\run.bat

# Select mode: 1 (CPU), 2 (CUDA), or 3 (Hybrid)
# App will download CLIP model (~500MB)
```

**Models downloaded on first run:**
- CLIP (ViT-B/32): ~500MB
- HuggingFace tokenizer cache
- Dependencies (if missing)

## Subsequent Runs (Offline)

```powershell
# Terminal 1: Start Ollama (local server)
ollama serve

# Terminal 2: Run the app (works offline!)
.\run.bat
```

## What Works Offline

✅ **Fully offline:**
- Live camera feed
- YOLO object detection
- CLIP visual embedding & matching
- Ollama LLM responses (local)
- Face recognition (OpenCV LBPH)
- Teaching/learning new objects
- Drawing rectangles/polygons
- Chat with local LLM

❌ **Requires internet:**
- First-time model download
- GitHub updates
- Downloading new Ollama models
- External API calls

## Pre-Flight Check

When you launch `run.bat`, it automatically checks:

```
1. Internet connection (detects offline mode)
2. Python dependencies (torch, cv2, PIL, numpy)
3. Ollama server running
4. Cached models available
```

**If checks fail:** See error messages for fixes.

## Offline Troubleshooting

### "Ollama not running"
```powershell
# In a new terminal
ollama serve
```

### "Models not cached"
- **First run only**: Connect to internet, run app once
- Models download automatically to: `~\.cache\hybrid-assistant\huggingface\`

### "Missing Python dependencies"
```powershell
pip install -r requirements.txt
```

### "Cannot download CLIP model"
- You're on first run and offline
- **Solution**: Connect to internet once, then you're offline-ready

## Model Cache

**Cache location:**
```
C:\Users\<YourUser>\.cache\hybrid-assistant\huggingface\
```

**What's cached:**
- CLIP model (ViT-B/32): ~500MB
- Tokenizers: ~50MB
- Config files: ~1MB

**Total size: ~500-600MB**

## Check Cache Status

Run startup check manually:
```powershell
python startup_check.py
```

Shows:
- Internet availability
- Python dependencies
- Ollama status
- Cached models size
- Available Ollama models

## Offline Performance

| Component | Requires Internet | Notes |
|-----------|------------------|-------|
| CLIP embeddings | No (after cache) | Runs on GPU/CPU locally |
| YOLO detection | No | Pre-installed |
| Ollama LLM | No | Local server only |
| Face recognition | No | OpenCV LBPH |
| Camera feed | No | Local webcam |
| Chat | No | Uses local Ollama |
| Teaching/Learning | No | Local storage |

## Advanced: Manual Offline Setup

If you prefer manual setup:

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download CLIP model manually (run once with internet)
python -c "import clip; clip.load('ViT-B/32')"

# Now you can run offline
python detection_gui.py
```

## Storage Requirements

After complete setup:
```
- CLIP model: 500MB
- Ollama models: 2-7GB (depends on which models you install)
- App cache: 50MB
- Learned data: ~100MB
---
Total: ~3-8GB
```

## Environment Variables (Advanced)

To force offline mode:
```powershell
set TRANSFORMERS_OFFLINE=1
python detection_gui.py
```

Or in Python:
```python
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

## FAQ

**Q: Can I use it without Ollama?**
A: No. Ollama is required for LLM responses. But it runs locally, so no internet needed after initial setup.

**Q: How do I know if I'm offline?**
A: Run `startup_check.py`. It will show "Running in offline mode" if no internet detected.

**Q: Can I switch between online/offline?**
A: Yes. The app auto-detects internet and adapts automatically.

**Q: What if models aren't cached?**
A: On first run with internet, models download automatically. Without internet on first run, you'll get an error (just run with internet once).

**Q: Can I use different Ollama models?**
A: Yes! The app works with any Ollama model. Just pull it first:
```powershell
ollama pull gemma2:2b
ollama pull mistral:latest
```

**Q: How do I update the app offline?**
A: Clone/update from GitHub when you have internet, then run offline.

---

**Summary**: ☑️ **After first run, your Hybrid Assistant works completely offline!**
