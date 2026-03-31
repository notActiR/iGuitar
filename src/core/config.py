"""配置管理模块"""
import json
import os
import shutil

class Config:
    DEFAULTS = {
        'camera': {'id': 0, 'width': 1280, 'height': 720},
        'hand_tracker': {'max_hands': 2, 'detection_confidence': 0.5, 'tracking_confidence': 0.5},
        'fretboard': {'strings': 6, 'frets': 12, 'distance_threshold': 30},
        'display': {'show_fps': True, 'show_skeleton': True, 'flip_horizontal': True}
    }

    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.data = self.load()

    def load(self):
        if not os.path.exists(self.config_path):
            template = 'config.template.json'
            if os.path.exists(template):
                shutil.copy(template, self.config_path)
                print(f"✓ 已创建配置文件: {self.config_path}")
            return self.DEFAULTS.copy()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ 配置文件读取失败: {e}，使用默认配置")
            return self.DEFAULTS.copy()

    def get(self, *keys, default=None):
        value = self.data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default
