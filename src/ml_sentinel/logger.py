import logging
import yaml
from pathlib import Path

def setup_logger(name: str = "ml_sentinel"):
    with open(Path(__file__).parent.parent.parent / "config/logging.yaml") as f:
        log_config = yaml.safe_load(f)
    logging.config.dictConfig(log_config)
    return logging.getLogger(name)
