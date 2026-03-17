"""
曲目管理模块：加载 JSON 曲目文件，提供事件切换和当前目标获取
"""
import json
from data.chord_db import CHORDS

class Song:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.title = data['title']
        self.tempo = data.get('tempo', 120)
        self.time_signature = data.get('time_signature', '4/4')

        # 支持两种格式：chords（和弦名列表）或 notes（具体弦品列表）
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
        """返回当前事件的预期指法字典 {弦号: 品号}"""
        if self.current_index >= len(self.events):
            return None  # 曲目结束

        ev = self.events[self.current_index]
        if self.type == 'chord':
            chord_name = ev['chord']
            # 从 chord_db 中获取标准指法
            return CHORDS.get(chord_name, {}).copy()
        else:
            # 单音：只有一根弦有品，其他为空弦
            target = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
            target[ev['string']] = ev['fret']
            return target

    def next(self):
        """切换到下一个事件，返回是否成功"""
        if self.current_index < len(self.events) - 1:
            self.current_index += 1
            return True
        return False

    def prev(self):
        """切换到上一个事件"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def reset(self):
        """重置到开头"""
        self.current_index = 0