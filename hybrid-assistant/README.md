## Note on DuckDuckGo Search Library

As of November 2025, the `ddgs` package (the successor to `duckduckgo-search`) could not be installed or imported in this environment. Therefore, the project continues to use `duckduckgo-search` for internet search features. If you see a deprecation warning, it is safe to ignore for now. Future updates may switch to `ddgs` if compatibility improves.
# Hybrid Assistant

A local, privacy-focused **face recognition and chatbot** application that combines camera-based visual detection with a local LLM backend. Teach the assistant to recognize faces and objects through the GUI or chat, get intelligent scene descriptions, and have conversations—all running locally and offline.

Developed by **@daqa020** with ❤️

## Features

- **Live Camera GUI** — Real-time video feed with object detection overlays and interactive chat panel
- **Visual Learning** — Teach new objects by drawing rectangles/polygons on the camera feed
- **Chat-Based Teaching** — Use simple commands like `Teach: object_name - description`
- **Face Recognition** — OpenCV LBPH-based face detection and recognition (optional)
- **Embeddings** — CLIP-based visual embeddings for robust object matching
- **Local LLM** — Ollama HTTP API integration for text generation
- **Privacy-First** — All data stored locally; no cloud dependency

## Quick Start

### Prerequisites

- Python 3.11+
- Camera (webcam or USB)
- Ollama installed and running (see [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup)

### ⚡ Getting Started (Windows)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarahubanic/hybrid-assistant.git
   cd hybrid-assistant
   ```

2. **That's it! Just double-click `run.bat`**
   - The script will automatically:
     - Create a virtual environment
     - Install all dependencies
     - Ask you to choose a mode (CPU / CUDA / Hybrid)
     - Start the app

3. **Start Ollama** (in a separate terminal/window):
   ```bash
   ollama serve
   ```

### Manual Setup (Advanced)

If you prefer manual control:
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python detection_gui.py
```

### Linux / macOS

```bash
git clone https://github.com/sarahubanic/hybrid-assistant.git
cd hybrid-assistant

# Create and activate venv
python -m venv venv
source venv/bin/activate

# Install and run
pip install -r requirements.txt
python detection_gui.py
```

## Usage

### Main GUI (`detection_gui.py`)

- **Camera View**: Live feed with detection overlays
- **Chat Panel**: Send messages or teach commands
- **Teach Dialog**: Open to draw rectangles/polygons around objects to learn

### Teaching an Object

1. Click **"Teach"** button to open the Teach dialog
2. Freeze the camera with the object visible
3. Draw a **rectangle** or **polygon** around the object
4. Add a name and optional description
5. Click **Save** — the object is now learned

### Chat Commands

- **`Teach: object_name - description`** — Teach the assistant using text only. Example:
   - `Teach: Umbrella - A collapsible black umbrella commonly stored in a bag`
   - The assistant will add this to the local knowledge base and use it for recognition and replies.

- **`teach:`** and **`Teach:`** are treated equivalently (case-insensitive) by the GUI.

- **`Search: query`** or **`pretrazi: query`** — Perform a DuckDuckGo search and return concise results (title, short snippet, URL).
   - Example: `Search: current weather in London`
   - Example (Serbian): `pretrazi: vreme u Beogradu`

- **`Add object` (GUI)** — Use the Teach dialog to visually add objects:
   1. Click **"Teach Me Something"**.
   2. Freeze the camera frame when the object is visible.
   3. Draw a **rectangle** or **polygon** around the object (or use face mode for faces).
   4. Enter a **Name** and **Description** and click **Save**. The object and its CLIP embedding will be stored in `learned_items/`.

- **`Teach:` (chat + image)** — For future improvements the app will support `Teach:` combined with an attached image or `Send + Img` button to teach from a crop; currently, use the Teach dialog for visual teaching.

## Files

| File | Purpose |
|------|---------|
| `detection_gui.py` | Main GUI application |
| `hybrid_assistant.py` | Core assistant logic |
| `mistral_chatbot.py` | Ollama chat integration |
| `ollama_client.py` | Ollama HTTP client |
| `video_assistant.py` | Video processing utilities |
| `requirements.txt` | Python dependencies |
| `INSTRUCTIONS.md` | Detailed setup & configuration |

## Configuration

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for:
- Ollama model recommendations
- Face recognition setup
- Visual embedding configuration
- Troubleshooting

## Project Structure

```
hybrid-assistant/
├── detection_gui.py          # Main GUI
├── hybrid_assistant.py        # Core logic
├── mistral_chatbot.py         # Chat backend
├── ollama_client.py           # LLM client
├── video_assistant.py         # Video utilities
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── INSTRUCTIONS.md            # Setup guide
├── LICENSE                    # MIT License
├── learned_items/             # Persistent knowledge
│   ├── knowledge.json         # Learned facts
│   └── visual_samples/        # Saved object crops
├── logs/                      # Run logs
└── models/                    # Model cache (optional)
```

## Privacy & Data

- ✅ Runs entirely on your machine
- ✅ No cloud calls or telemetry
- ✅ All learned data stored locally in `learned_items/`
- ✅ Optional face recognition (local only)

## License

MIT License — See [LICENSE](LICENSE) for details.

## 🙏 Special Thanks

This project was made possible thanks to:

- **[@jinnosux](https://github.com/jinnosux)** — For providing the hardware tools and the original idea that inspired this project. Your vision made this possible! 🚀

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Last Updated**: November 17, 2025
