"""配置管理模块"""
import json
import os

class Config:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.data = self.load()

    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get(self, *keys, default=None):
        value = self.data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default
