import logging
import os
import yaml


def setup_logger(name: str = "ml-sentinel", config_path: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        level = getattr(logging, config.get("root", {}).get("level", "INFO"))
        logger.setLevel(level)
        for handler_name in config.get("root", {}).get("handlers", []):
            handler_config = config.get("handlers", {}).get(handler_name, {})
            handler_class = handler_config.get("class", "logging.StreamHandler")
            if handler_class == "logging.StreamHandler":
                handler = logging.StreamHandler()
            elif handler_class == "logging.FileHandler":
                handler = logging.FileHandler(handler_config.get("filename", "app.log"))
            else:
                handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, handler_config.get("level", "INFO")))
            formatter = logging.Formatter(
                config.get("formatters", {}).get("standard", {}).get(
                    "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    return logger
