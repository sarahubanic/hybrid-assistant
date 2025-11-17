#!/usr/bin/env python3
"""Quick test script to verify Ollama connectivity and model responses."""

import requests
import json

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"
MODELS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

def test_connection():
    """Test basic connection to Ollama."""
    print("[TEST] Checking Ollama connectivity...")
    try:
        resp = requests.get(MODELS_ENDPOINT, timeout=5)
        if resp.status_code == 200:
            print(f"[OK] Ollama is reachable at {OLLAMA_BASE_URL}")
            return True
        else:
            print(f"[FAIL] Ollama returned status {resp.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Cannot reach Ollama: {e}")
        return False

def list_models():
    """List available Ollama models."""
    print("\n[TEST] Listing available models...")
    try:
        resp = requests.get(MODELS_ENDPOINT, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('models', [])
            print(f"[OK] Found {len(models)} model(s):")
            for model in models:
                name = model.get('name', 'unknown')
                size = model.get('size', 0)
                size_gb = size / (1024**3)
                print(f"  - {name} ({size_gb:.2f} GB)")
            return models
        else:
            print(f"[FAIL] Status {resp.status_code}")
            return []
    except Exception as e:
        print(f"[FAIL] Error listing models: {e}")
        return []

def test_generate(model_name="mistral:latest"):
    """Test text generation with a model."""
    print(f"\n[TEST] Testing generation with {model_name}...")
    try:
        payload = {
            "model": model_name,
            "prompt": "Hello, how are you?",
            "stream": False,
            "max_tokens": 50,
            "temperature": 0.2,
        }
        resp = requests.post(GENERATE_ENDPOINT, json=payload, timeout=120)
        if resp.status_code == 200:
            # Ollama returns line-delimited JSON
            text_parts = []
            for line in resp.text.strip().split('\n'):
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if 'response' in j:
                        text_parts.append(j['response'])
                    if j.get('done', False):
                        break
                except Exception as e:
                    pass
            
            text = ''.join(text_parts)
            print(f"[OK] Generation successful:")
            print(f"  Response: {text[:100]}...")
            return True
        else:
            print(f"[FAIL] Status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Generation error: {e}")
        return False

def test_embed(model_name="nomic-embed-text:latest"):
    """Test embedding with a model."""
    print(f"\n[TEST] Testing embedding with {model_name}...")
    try:
        payload = {
            "model": model_name,
            "input": "This is a test sentence for embeddings.",
        }
        resp = requests.post(EMBED_ENDPOINT, json=payload, timeout=120)
        if resp.status_code == 200:
            j = resp.json()
            if 'embedding' in j:
                embedding = j['embedding']
                print(f"[OK] Embedding successful:")
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
            elif 'embeddings' in j and len(j['embeddings']) > 0:
                embedding = j['embeddings'][0]
                print(f"[OK] Embedding (from list) successful:")
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
            else:
                print(f"[OK] Response: {j}")
            return True
        else:
            print(f"[FAIL] Status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Embedding error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Ollama Connection & Model Test")
    print("=" * 60)
    
    # Test connectivity
    if not test_connection():
        print("\n[ERROR] Ollama is not reachable. Make sure it's running:")
        print("  1. Start Ollama desktop app, or")
        print("  2. Run 'ollama serve' in a terminal")
        exit(1)
    
    # List models
    models = list_models()
    if not models:
        print("\n[ERROR] No models found. Install one with 'ollama pull <model-name>'")
        exit(1)
    
    # Test generate with mistral
    test_generate("mistral:latest")
    
    # Test embedding with nomic-embed-text
    test_embed("nomic-embed-text:latest")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
