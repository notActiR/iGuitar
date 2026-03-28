"""WebUI 界面 - 基于 Gradio"""
import gradio as gr
import cv2
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core import Camera, HandTracker, VideoPreprocessor
from src.ui.display import Display
from src.mapping import FretboardMapper
from src.data import Song

class iGuitarWebUI:
    def __init__(self):
        self.camera = None
        self.hand_tracker = None
        self.preprocessor = None
        self.display = None
        self.mapper = None
        self.song = None
        self.current_target = None

    def initialize(self):
        try:
            if not os.path.exists('calibration_matrix.npy'):
                return "⚠️ 请先运行标定程序"

            self.camera = Camera(camera_id=0)
            self.hand_tracker = HandTracker(max_hands=2)
            self.preprocessor = VideoPreprocessor(flip=True, target_width=640)
            self.display = Display()
            self.mapper = FretboardMapper('calibration_matrix.npy')
            return "✅ 初始化成功"
        except Exception as e:
            return f"❌ 初始化失败: {e}"

    def load_song(self, song_file):
        try:
            self.song = Song(f'assets/songs/{song_file}')
            self.current_target = self.song.get_current_target()
            return f"✅ 已加载: {self.song.title}"
        except Exception as e:
            return f"❌ 加载失败: {e}"

    def get_frame(self):
        if not self.camera:
            return None

        ret, frame = self.camera.read_frame()
        if not ret:
            return None

        rgb_frame, bgr_frame = self.preprocessor.process(frame)
        results = self.hand_tracker.detect(rgb_frame)
        output_frame = self.display.draw_landmarks(bgr_frame, results)

        if results.hand_landmarks and self.song:
            hand_landmarks = results.hand_landmarks[0]
            actual_fingers = self.mapper.get_finger_frets(hand_landmarks, output_frame.shape)

            for finger, pos in actual_fingers.items():
                if pos is None:
                    continue
                fret, string = pos
                idx = self.mapper.FINGER_INDICES[finger]
                lm = hand_landmarks[idx]
                h, w, _ = output_frame.shape
                x, y = int(lm.x * w), int(lm.y * h)

                expected_fret = self.current_target.get(string, 0)
                color = (0, 255, 0) if expected_fret == fret else (0, 0, 255)
                cv2.circle(output_frame, (x, y), 10, color, -1)

        return cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

def create_ui():
    app = iGuitarWebUI()

    with gr.Blocks(title="🎸 iGuitar WebUI") as demo:
        gr.Markdown("# 🎸 iGuitar - 吉他学习辅助系统")

        with gr.Row():
            with gr.Column():
                init_btn = gr.Button("初始化系统", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)

                song_dropdown = gr.Dropdown(
                    choices=["twinkle.json", "simple_practice.json"],
                    label="选择歌曲",
                    value="twinkle.json"
                )
                load_btn = gr.Button("加载歌曲")
                song_status = gr.Textbox(label="歌曲状态", interactive=False)

            with gr.Column():
                video = gr.Image(label="摄像头画面")

        init_btn.click(app.initialize, outputs=[status])
        load_btn.click(app.load_song, inputs=[song_dropdown], outputs=[song_status])

        demo.load(lambda: None, None, video, every=0.1).then(app.get_frame, outputs=[video])

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)
