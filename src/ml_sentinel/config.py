import yaml
from pathlib import Path

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self._load_configs()

    def _load_configs(self):
        with open(self.base_dir / "config/drift_config.yaml") as f:
            self.drift = yaml.safe_load(f)["drift"]
        with open(self.base_dir / "config/model_promotion_policy.json") as f:
            import json
            self.promotion = json.load(f)

config = Config()
