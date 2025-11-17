#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from offline_handler import offline_manager

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_python_deps():
    """Check if core Python dependencies are installed"""
    required = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    return len(missing) == 0, missing

def check_prerequisites():
    """Check all prerequisites before launch"""
    print_header("🔍 HYBRID ASSISTANT - PRE-FLIGHT CHECKS")
    
    all_passed = True
    
    # 1. Internet check
    print("1️⃣  Checking internet connection...")
    internet = offline_manager.check_internet()
    if internet:
        print("   ✅ Internet available\n")
    else:
        print("   ⚠️  Running in offline mode\n")
    
    # 2. Python dependencies
    print("2️⃣  Checking Python dependencies...")
    deps_ok, missing = check_python_deps()
    if deps_ok:
        print("   ✅ All core dependencies installed\n")
    else:
        print(f"   ❌ Missing: {', '.join(missing)}")
        print("   📌 Run: pip install -r requirements.txt\n")
        all_passed = False
    
    # 3. Ollama check
    print("3️⃣  Checking Ollama server...")
    ollama_running = offline_manager.verify_ollama()
    if ollama_running:
        print("   ✅ Ollama running at http://localhost:11434")
        models = offline_manager.get_ollama_models()
        if models:
            print(f"   📦 Available models: {', '.join(models[:3])}")
            if len(models) > 3:
                print(f"      ... and {len(models) - 3} more\n")
            else:
                print()
        else:
            print("   ⚠️  No models available\n")
    else:
        print("   ❌ Ollama NOT running!")
        print("   📌 Start Ollama with: ollama serve\n")
        all_passed = False
    
    # 4. Model cache check
    print("4️⃣  Checking cached models...")
    cache_dir = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
    cache_size = offline_manager.get_cache_size()
    
    if cache_dir.exists() and list(cache_dir.glob('**/*')):
        print(f"   ✅ Models cached ({cache_size})\n")
    else:
        if internet:
            print("   ⓘ Models not cached (will download on first run)\n")
        else:
            print("   ⚠️  Models not cached and no internet available!")
            print("   📌 First run requires internet to download models\n")
            all_passed = False
    
    # Summary
    print_header("✅ PRE-FLIGHT CHECK COMPLETE")
    
    if all_passed:
        print("✨ All checks passed! Ready to launch.\n")
        return 0
    else:
        print("⚠️  Some checks failed. See above for details.\n")
        return 1

if __name__ == "__main__":
    exit_code = check_prerequisites()
    sys.exit(exit_code)
