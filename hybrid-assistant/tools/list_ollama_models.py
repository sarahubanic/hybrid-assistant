import requests
import json

URL = "http://127.0.0.1:11434/api/models"

try:
    resp = requests.get(URL, timeout=5)
    resp.raise_for_status()
    models = resp.json()
    print(json.dumps(models, indent=2))
except Exception as e:
    print(f"Error contacting Ollama API at {URL}: {e}")
    try:
        print("Raw response:", resp.text)
    except:
        pass
