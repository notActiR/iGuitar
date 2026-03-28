"""练习统计模块"""
import json
import os
from datetime import datetime

class PracticeStats:
    def __init__(self, stats_file='practice_stats.json'):
        self.stats_file = stats_file
        self.data = self.load()
        self.current_session = {
            'start_time': datetime.now().isoformat(),
            'correct': 0,
            'wrong': 0,
            'total': 0
        }

    def load(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'sessions': [], 'total_practice_time': 0}

    def record_attempt(self, is_correct):
        self.current_session['total'] += 1
        if is_correct:
            self.current_session['correct'] += 1
        else:
            self.current_session['wrong'] += 1

    def get_accuracy(self):
        if self.current_session['total'] == 0:
            return 0
        return (self.current_session['correct'] / self.current_session['total']) * 100

    def save_session(self):
        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['accuracy'] = self.get_accuracy()
        self.data['sessions'].append(self.current_session)

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
