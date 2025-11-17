"""
Simple GGUF inspection utility.

Usage:
  python tools/gguf_inspect.py /path/to/model.gguf

This prints the file name (stem) and attempts to show any metadata fields that contain 'name'.
"""
import sys
from pathlib import Path

try:
    import gguf
    GGUF_PY_AVAILABLE = True
except Exception:
    GGUF_PY_AVAILABLE = False


def inspect(path: str):
    p = Path(path)
    if not p.exists():
        print('File not found:', path)
        return
    print('File:', p.name)
    print('Model id (from filename):', p.stem)
    if GGUF_PY_AVAILABLE:
        try:
            reader = gguf.GGUFReader(str(p))
            print('GGUF reader fields:', list(reader.fields.keys()))
            name_keys = [k for k in reader.fields.keys() if 'name' in k.lower()]
            for k in name_keys:
                field = reader.fields[k]
                print('Field:', k)
                try:
                    if hasattr(field, 'parts') and len(field.parts) > 0:
                        part = field.parts[0]
                        if isinstance(part, (bytes, bytearray)):
                            print(' ->', part.decode('utf-8','ignore'))
                        else:
                            print(' ->', part)
                except Exception as e:
                    print('  (failed to decode part):', e)
        except Exception as e:
            print('Failed to open GGUF with gguf-py:', e)
    else:
        print('gguf-py not installed; install it to inspect metadata.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/gguf_inspect.py /path/to/model.gguf')
    else:
        inspect(sys.argv[1])
