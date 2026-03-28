"""WebUI 界面 - 基于 Gradio"""
import gradio as gr
import cv2
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core import Camera, HandTracker, VideoPreprocessor
from src.ui.display import Display
from src.mapping import FretboardMapper
from src.data import Song, CHORDS, PracticeStats

class iGuitarWebUI:
    def __init__(self):
        self.camera = None
        self.hand_tracker = None
        self.preprocessor = None
        self.display = None
        self.mapper = None
        self.song = None
        self.current_target = None
        self.stats = PracticeStats()
        self.initialized = False

    def initialize(self):
        try:
            if not os.path.exists('calibration_matrix.npy'):
                return "⚠️ 请先运行标定程序", ""

            self.camera = Camera(camera_id=0)
            self.hand_tracker = HandTracker(max_hands=2)
            self.preprocessor = VideoPreprocessor(flip=True, target_width=640)
            self.display = Display()
            self.mapper = FretboardMapper('calibration_matrix.npy')
            self.initialized = True
            return "✅ 初始化成功", "系统已就绪"
        except Exception as e:
            return f"❌ 初始化失败: {e}", ""

    def load_song(self, song_file):
        try:
            self.song = Song(f'assets/songs/{song_file}')
            self.current_target = self.song.get_current_target()
            return f"✅ 已加载: {self.song.title}", self.get_chord_info()
        except Exception as e:
            return f"❌ 加载失败: {e}", ""

    def get_chord_info(self):
        if not self.song:
            return ""
        ev = self.song.events[self.song.current_index]
        if self.song.type == 'chord':
            return f"当前和弦: {ev['chord']} ({self.song.current_index+1}/{len(self.song.events)})"
        return f"音符: 弦{ev['string']} 品{ev['fret']}"

    def next_chord(self):
        if self.song and self.song.next():
            self.current_target = self.song.get_current_target()
            return self.get_chord_info()
        return "已到最后"

    def prev_chord(self):
        if self.song and self.song.prev():
            self.current_target = self.song.get_current_target()
            return self.get_chord_info()
        return "已到开头"

    def get_frame(self):
        if not self.camera or not self.initialized:
            return None, "等待初始化..."

        ret, frame = self.camera.read_frame()
        if not ret:
            return None, "读取失败"

        rgb_frame, bgr_frame = self.preprocessor.process(frame)
        results = self.hand_tracker.detect(rgb_frame)
        output_frame = self.display.draw_landmarks(bgr_frame, results)

        stats_text = f"FPS: 30 | 手部: {len(results.hand_landmarks) if results.hand_landmarks else 0}"

        if results.hand_landmarks and self.song:
            hand_landmarks = results.hand_landmarks[0]
            actual_fingers = self.mapper.get_finger_frets(hand_landmarks, output_frame.shape)

            correct = 0
            total = 0

            for finger, pos in actual_fingers.items():
                if pos is None:
                    continue
                fret, string = pos
                idx = self.mapper.FINGER_INDICES[finger]
                lm = hand_landmarks[idx]
                h, w, _ = output_frame.shape
                x, y = int(lm.x * w), int(lm.y * h)

                expected_fret = self.current_target.get(string, 0)
                is_correct = expected_fret == fret
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.circle(output_frame, (x, y), 10, color, -1)

                if expected_fret != 0:
                    total += 1
                    if is_correct:
                        correct += 1

            if total > 0:
                acc = (correct / total) * 100
                stats_text += f" | 正确率: {acc:.0f}%"

        return cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), stats_text

def create_ui():
    app = iGuitarWebUI()

    with gr.Blocks(title="🎸 iGuitar WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎸 iGuitar - 吉他学习辅助系统")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 系统控制")
                init_btn = gr.Button("🚀 初始化系统", variant="primary", size="lg")
                status = gr.Textbox(label="状态", interactive=False)
                info = gr.Textbox(label="信息", interactive=False)

                gr.Markdown("### 歌曲选择")
                song_dropdown = gr.Dropdown(
                    choices=["twinkle.json", "simple_practice.json"],
                    label="选择歌曲",
                    value="twinkle.json"
                )
                load_btn = gr.Button("📂 加载歌曲", variant="secondary")
                song_status = gr.Textbox(label="歌曲状态", interactive=False)

                gr.Markdown("### 练习控制")
                with gr.Row():
                    prev_btn = gr.Button("⬅️ 上一个")
                    next_btn = gr.Button("➡️ 下一个")
                chord_info = gr.Textbox(label="当前和弦", interactive=False)

                stats = gr.Textbox(label="实时统计", interactive=False)

            with gr.Column(scale=2):
                video = gr.Image(label="摄像头画面", height=480)

        init_btn.click(app.initialize, outputs=[status, info])
        load_btn.click(app.load_song, inputs=[song_dropdown], outputs=[song_status, chord_info])
        next_btn.click(app.next_chord, outputs=[chord_info])
        prev_btn.click(app.prev_chord, outputs=[chord_info])

        timer = gr.Timer(0.01)
        timer.tick(app.get_frame, outputs=[video, stats])

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)
