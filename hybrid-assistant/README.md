# Hybrid Assistant

A local, privacy-focused hybrid assistant that combines camera-based visual detection with a local LLM backend. Teach the assistant new objects through the GUI or chat, and get intelligent scene descriptions and object recognition—all running locally.

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
- Ollama running locally (see [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup)

### Installation

```bash
# Clone the repo
git clone https://github.com/sarahubanic/hybrid-assistant.git
cd hybrid-assistant

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Assistant

```bash
# Start Ollama (in a separate terminal)
ollama serve

# In another terminal, run the GUI
python detection_gui.py
```

Or use the provided batch script:
```bash
./start_assistant.bat  # Windows
./start_assistant.sh   # Linux/Mac
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

- **`Teach: object_name - description`** — Teach from text
- Regular chat — Ask questions about what the camera sees

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Last Updated**: November 17, 2025
