import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")


def load_config(filename: str) -> dict:
    path = os.path.join(CONFIG_DIR, filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)
