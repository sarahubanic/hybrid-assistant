"""
Offline Mode Handler for Hybrid Assistant
Manages model caching and offline fallbacks
"""

import os
import sys
from pathlib import Path
import json
from typing import Optional, Dict
import datetime

class OfflineManager:
    """Manages offline mode and model caching"""
    
    def __init__(self):
        self.cache_dir = Path.home() / '.cache' / 'hybrid-assistant'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.cache_dir / 'offline_status.json'
        self.setup_offline_environment()
        
    def setup_offline_environment(self):
        """Configure environment for offline use"""
        # Configure cache/home locations
        os.environ['HF_HOME'] = str(self.cache_dir / 'huggingface')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Create subdirectories
        os.makedirs(os.environ['HF_HOME'], exist_ok=True)

        # Detect internet connectivity and force TRANSFORMERS_OFFLINE when offline
        try:
            online = self.check_internet()
        except Exception:
            online = False

        if online:
            # Respect any existing setting, default to '0' (online)
            os.environ['TRANSFORMERS_OFFLINE'] = os.environ.get('TRANSFORMERS_OFFLINE', '0')
        else:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            self.log_offline_status("No internet connection detected.")
            # Persist the offline status for diagnostics
            try:
                self.log_status({'offline': True})
            except Exception:
                pass
    
    def log_offline_status(self, reason: str):
        """Log offline status with timestamp and reason"""
        status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason
        }
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=4)

    def check_internet(self) -> bool:
        """Check internet connectivity by pinging a reliable server"""
        import subprocess
        try:
            subprocess.check_call(['ping', '-n', '1', '8.8.8.8'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def verify_ollama(self, host="http://localhost:11434") -> bool:
        """Verify Ollama is running"""
        try:
            import requests
            response = requests.get(f"{host}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception as e:
            return False
    
    def get_ollama_models(self, host="http://localhost:11434") -> list:
        """Get list of available Ollama models"""
        try:
            import requests
            response = requests.get(f"{host}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            pass
        return []
    
    def log_status(self, status: Dict):
        """Log offline status"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save status: {e}")
    
    def get_status(self) -> Dict:
        """Get offline status"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def get_cache_size(self) -> str:
        """Get total size of cached models"""
        try:
            total = 0
            hf_cache = Path(os.environ['HF_HOME'])
            if hf_cache.exists():
                for f in hf_cache.rglob('*'):
                    if f.is_file():
                        total += f.stat().st_size
            
            # Convert to GB
            gb = total / (1024 ** 3)
            return f"{gb:.2f} GB"
        except Exception:
            return "Unknown"

# Global instance
offline_manager = OfflineManager()
