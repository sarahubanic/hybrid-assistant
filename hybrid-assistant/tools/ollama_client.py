"""Simple Ollama HTTP client helper used by the project.

Usage examples:
  python tools/ollama_client.py --prompt "Say hello" --out out.txt
  python tools/ollama_client.py --prompt-file prompts.txt --out reply.txt

This uses the local Ollama HTTP server at http://127.0.0.1:11434 by default.
"""
import argparse
import json
import os
import requests
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")


def generate(prompt: str, model: str = "mistral:latest", max_tokens: int = 256, timeout: int = 120):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, help="Prompt text")
    p.add_argument("--prompt-file", type=Path, help="File with prompt text")
    p.add_argument("--model", type=str, default="mistral:latest")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", type=Path, default=None, help="Write raw response to file")
    args = p.parse_args()

    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt
    else:
        raise SystemExit("Provide --prompt or --prompt-file")

    text = generate(prompt, model=args.model, max_tokens=args.max_tokens)
    # Save raw response (which may be streamed JSON chunks) to file or print
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote reply to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
