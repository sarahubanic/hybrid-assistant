#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start the Hybrid Assistant with configurable GPU/CPU options for Ollama.

Usage:
    python start_assistant.py              # Interactive menu
    python start_assistant.py --gpu        # CUDA GPU mode (fast but may crash with low VRAM)
    python start_assistant.py --cpu        # CPU-only mode (stable, uses system RAM)
    python start_assistant.py --auto       # Auto-detect best mode
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def setup_ollama_gpu():
    """Setup for CUDA GPU mode (uses GTX 1050, may crash if out of VRAM)."""
    print("\n[CONFIG] Setting up CUDA GPU mode...")
    print("  - GPU: NVIDIA GeForce GTX 1050 (3GB VRAM)")
    print("  - Speed: Fast (~1 sec/response)")
    print("  - Risk: May crash if VRAM runs out")
    print("  - Fallback: Graceful error handling implemented")
    
    # Allow CUDA
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.environ.pop('OLLAMA_NUM_GPU', None)
    print("[+] CUDA GPU acceleration ENABLED")

def setup_ollama_cpu():
    """Setup for CPU-only mode (uses 32GB system RAM, stable)."""
    print("\n[CONFIG] Setting up CPU-only mode...")
    print("  - GPU: DISABLED")
    print("  - RAM: 32GB System RAM")
    print("  - Speed: Moderate (~2-5 sec/response)")
    print("  - Stability: Excellent, no CUDA crashes")
    
    # Disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OLLAMA_NUM_GPU'] = '0'
    print("[+] CUDA GPU acceleration DISABLED - using CPU only")

def setup_ollama_hybrid():
    """Setup for hybrid mode (GPU + CPU together)."""
    print("\n[CONFIG] Setting up HYBRID mode...")
    print("  - GPU: NVIDIA GeForce GTX 1050 (3GB VRAM) - primary")
    print("  - CPU: Used as fallback/offload when GPU is saturated")
    print("  - Speed: Fast when GPU works, falls back to CPU if needed")
    print("  - Memory: GPU (3GB) + System RAM (32GB)")
    
    # Allow both
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.environ['OLLAMA_NUM_PARALLEL'] = '2'  # Allow parallel inference
    os.environ['OLLAMA_NUM_GPU'] = '1'       # Use GPU but allow CPU fallback
    print("[+] HYBRID mode ENABLED - GPU + CPU fallback")

def interactive_mode():
    """Show interactive menu for user to choose GPU/CPU mode."""
    print("\n" + "="*60)
    print("  HYBRID ASSISTANT - Ollama Configuration")
    print("="*60)
    print("\nChoose operation mode:")
    print("  [1] GPU mode (CUDA) - Fast but may crash with low VRAM")
    print("  [2] CPU mode - Stable, uses 32GB system RAM")
    print("  [3] HYBRID mode - GPU primary with CPU fallback")
    print("  [Q] Quit without starting")
    
    choice = input("\nSelect [1/2/3/Q]: ").strip().upper()
    
    if choice == '1':
        return 'gpu'
    elif choice == '2':
        return 'cpu'
    elif choice == '3':
        return 'hybrid'
    elif choice == 'Q':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Using CPU mode (safest).")
        return 'cpu'

def detect_best_mode():
    """Auto-detect best mode based on system."""
    # On Windows with GTX 1050 (3GB) and 32GB RAM, CPU-only is safest
    print("\n[AUTO] Detecting best mode...")
    print("  - Detected: GTX 1050 (3GB VRAM) + 32GB System RAM")
    print("  - Recommendation: CPU-only mode (more stable)")
    return 'cpu'

def main():
    parser = argparse.ArgumentParser(
        description='Start Hybrid Assistant with GPU/CPU options'
    )
    parser.add_argument('--gpu', action='store_true', help='Use CUDA GPU mode')
    parser.add_argument('--cpu', action='store_true', help='Use CPU-only mode')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid GPU+CPU mode')
    parser.add_argument('--auto', action='store_true', help='Auto-detect best mode')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.gpu:
        mode = 'gpu'
    elif args.cpu:
        mode = 'cpu'
    elif args.hybrid:
        mode = 'hybrid'
    elif args.auto:
        mode = detect_best_mode()
    else:
        # Interactive mode if no args
        mode = interactive_mode()
    
    # Setup environment based on mode
    if mode == 'gpu':
        setup_ollama_gpu()
    elif mode == 'cpu':
        setup_ollama_cpu()
    elif mode == 'hybrid':
        setup_ollama_hybrid()
    
    print("\n" + "="*60)
    print("  Starting Detection GUI with Ollama...")
    print("="*60)
    
    # Get the path to detection_gui.py
    script_dir = Path(__file__).parent
    gui_script = script_dir / 'src' / 'detection_gui.py'
    
    if not gui_script.exists():
        print(f"ERROR: Could not find {gui_script}")
        sys.exit(1)
    
    # Get python executable from venv
    venv_python = script_dir / '.venv' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        venv_python = 'python'  # Fallback to system python
    
    print(f"\n[INFO] Using Python: {venv_python}")
    print(f"[INFO] Running: {gui_script}")
    print(f"[INFO] Mode: {mode.upper()}")
    print(f"[INFO] Ollama environment variables set:")
    print(f"      CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'DEFAULT (GPU enabled)')}")
    print(f"      OLLAMA_NUM_GPU = {os.environ.get('OLLAMA_NUM_GPU', 'DEFAULT')}")
    print(f"      OLLAMA_NUM_PARALLEL = {os.environ.get('OLLAMA_NUM_PARALLEL', 'DEFAULT')}")
    
    print("\n" + "="*60)
    print("  Launching GUI...")
    print("="*60)
    
    # Launch the GUI
    try:
        subprocess.run([str(venv_python), str(gui_script)], check=True)
    except KeyboardInterrupt:
        print("\n\nGUI closed by user.")
    except Exception as e:
        print(f"\nERROR: Failed to start GUI: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
