# utils/config_loader.py

import json
from pathlib import Path

def load_config(config_path="config.json"):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)
