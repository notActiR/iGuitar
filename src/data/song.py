"""曲目管理模块"""
import json
from .chord_db import CHORDS

class Song:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.title = data['title']
        self.tempo = data.get('tempo', 120)
        self.time_signature = data.get('time_signature', '4/4')

        if 'chords' in data:
            self.events = data['chords']
            self.type = 'chord'
        elif 'notes' in data:
            self.events = data['notes']
            self.type = 'note'
        else:
            raise ValueError("曲目文件必须包含 chords 或 notes 字段")

        self.current_index = 0

    def get_current_target(self):
        if self.current_index >= len(self.events):
            return None

        ev = self.events[self.current_index]
        if self.type == 'chord':
            chord_name = ev['chord']
            return CHORDS.get(chord_name, {}).copy()
        else:
            target = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
            target[ev['string']] = ev['fret']
            return target

    def next(self):
        if self.current_index < len(self.events) - 1:
            self.current_index += 1
            return True
        return False

    def prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def reset(self):
        self.current_index = 0