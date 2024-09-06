__all__ = [
    'load'
]

# System imports
from pathlib import Path
import yaml

# Attempt to load config from the YAML file with the same name as this script
def load(file):
    try:
        with open(f"{Path(file).stem}.yml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
    return config
