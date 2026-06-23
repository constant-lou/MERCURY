from pathlib import Path

import yaml


class ConfigWrapper:
    def __init__(self, config_path):
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_id(self, record):
        return record.get(self.get("id_key", "id"))
