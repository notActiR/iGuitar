"""WebUI 界面 - 基于 Gradio (增强版，优化布局)"""
import gradio as gr
import cv2
import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core import Camera, HandTracker, VideoPreprocessor
from src.ui.display import Display
from src.mapping import FretboardMapper
from src.data import Song, CHORDS, PracticeStats

# ---------- 绘图函数（与 main.py 保持一致） ----------
def draw_target_points(frame, mapper, target, alpha=0.3):
    """绘制半透明绿色目标点"""
    overlay = frame.copy()
    for string, fret in target.items():
        if fret == 0:
            continue
        x, y = mapper.fretboard_to_pixel(fret, string)
        cv2.circle(overlay, (x, y), 12, (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def draw_missing_markers(frame, mapper, target, actual_fingers):
    """绘制缺失的目标点（灰色空心圆）"""
    missing_strings = []
    for string, fret in target.items():
        if fret == 0:
            continue
        found = False
        for pos in actual_fingers.values():
            if pos is not None and pos[1] == string and target.get(string, 0) == pos[0]:
                found = True
                break
        if not found:
            missing_strings.append(string)
    for string in missing_strings:
        fret = target[string]
        x, y = mapper.fretboard_to_pixel(fret, string)
        cv2.circle(frame, (x, y), 14, (128, 128, 128), 2)
    return frame

def draw_fretboard_diagram(frame, target, position='top-right', size=(200, 300), alpha=0.5):
    """绘制指板图（半透明，6弦在上）"""
    h, w = frame.shape[:2]
    diagram_w, diagram_h = size
    if position == 'top-right':
        x_start = w - diagram_w - 10
        y_start = 10
    else:
        x_start = w - diagram_w - 10
        y_start = h - diagram_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + diagram_w, y_start + diagram_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    string_spacing = diagram_h / 7
    for i in range(6):
        y = int(y_start + 20 + i * string_spacing)
        cv2.line(frame, (x_start + 20, y), (x_start + diagram_w - 20, y), (255, 255, 255), 1)

    fret_spacing = (diagram_w - 40) / 12
    for i in range(13):
        x = int(x_start + 20 + i * fret_spacing)
        cv2.line(frame, (x, y_start + 20), (x, y_start + diagram_h - 20), (255, 255, 255), 1)

    for string, fret in target.items():
        if fret == 0:
            continue
        string_idx = 6 - string
        y = int(y_start + 20 + string_idx * string_spacing + string_spacing / 2)
        x = int(x_start + 20 + (fret - 1) * fret_spacing + fret_spacing / 2)
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)

    for i in range(6):
        y = int(y_start + 20 + i * string_spacing + string_spacing / 2)
        cv2.putText(frame, str(6 - i), (x_start + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return frame

def get_feedback_text(chord_result):
    """生成英文评价文字"""
    errors = []
    for string, status in chord_result.items():
        if status == 'wrong':
            errors.append(f"String {string}: wrong fret")
        elif status == 'extra':
            errors.append(f"String {string}: extra finger")
        elif status == 'missing':
            errors.append(f"String {string}: missing")
    if errors:
        return " | ".join(errors)
    else:
        return "✓ Correct!"

# ---------- WebUI 主类 ----------
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
        self.show_target = True      # 显示半透明目标点
        self.show_diagram = True     # 显示指板图

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

    def refresh_songs(self):
        """扫描 assets/songs/ 目录，返回下拉选项"""
        songs_dir = 'assets/songs'
        if not os.path.exists(songs_dir):
            return []
        files = [f for f in os.listdir(songs_dir) if f.endswith('.json')]
        return gr.Dropdown(choices=files, value=files[0] if files else None)

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

    def toggle_target(self):
        self.show_target = not self.show_target
        return f"目标点: {'开启' if self.show_target else '关闭'}"

    def toggle_diagram(self):
        self.show_diagram = not self.show_diagram
        return f"指板图: {'开启' if self.show_diagram else '关闭'}"

    def get_frame(self):
        if not self.camera or not self.initialized:
            return None, "等待初始化...", ""

        ret, frame = self.camera.read_frame()
        if not ret:
            return None, "读取失败", ""

        rgb_frame, bgr_frame = self.preprocessor.process(frame)
        results = self.hand_tracker.detect(rgb_frame)
        output_frame = self.display.draw_landmarks(bgr_frame, results)

        stats_text = f"FPS: 30 | 手部: {len(results.hand_landmarks) if results.hand_landmarks else 0}"
        feedback_text = ""

        if results.hand_landmarks and self.song:
            hand_landmarks = results.hand_landmarks[0]
            actual_fingers = self.mapper.get_finger_frets(hand_landmarks, output_frame.shape)

            # 获取指尖像素坐标
            fingertip_pixels = {}
            finger_indices = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'little':20}
            for finger, idx in finger_indices.items():
                lm = hand_landmarks[idx]
                h, w, _ = output_frame.shape
                x = int(lm.x * w)
                y = int(lm.y * h)
                fingertip_pixels[finger] = (x, y)

            # 比对
            chord_result = {string: 'unknown' for string in range(1, 7)}
            for finger, pos in actual_fingers.items():
                if pos is None:
                    continue
                fret, string = pos
                expected_fret = self.current_target.get(string, 0)
                if expected_fret == fret:
                    chord_result[string] = 'correct'
                else:
                    if expected_fret != 0:
                        chord_result[string] = 'wrong'
                    else:
                        chord_result[string] = 'extra'

            for string, expected_fret in self.current_target.items():
                if expected_fret != 0:
                    found = any(pos is not None and pos[1] == string for pos in actual_fingers.values())
                    if not found:
                        chord_result[string] = 'missing'

            # 绘制目标点（半透明）
            if self.show_target:
                output_frame = draw_target_points(output_frame, self.mapper, self.current_target)

            # 绘制缺失标记
            output_frame = draw_missing_markers(output_frame, self.mapper, self.current_target, actual_fingers)

            # 绘制指尖圆点 + 连线
            for finger, pos in actual_fingers.items():
                if pos is None:
                    continue
                fret, string = pos
                expected_fret = self.current_target.get(string, 0)
                status = chord_result.get(string, 'unknown')
                if status == 'correct':
                    color = (0, 255, 0)
                elif status == 'wrong':
                    color = (0, 0, 255)
                elif status == 'extra':
                    color = (0, 165, 255)
                else:
                    color = (255, 255, 255)

                x, y = fingertip_pixels[finger]
                cv2.circle(output_frame, (x, y), 10, color, -1)
                cv2.circle(output_frame, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(output_frame, f"{fret},{string}", (x+15, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                if expected_fret != 0:
                    target_x, target_y = self.mapper.fretboard_to_pixel(expected_fret, string)
                    cv2.line(output_frame, (x, y), (target_x, target_y), (255, 255, 255), 2)

            # 生成评价文字
            feedback_text = get_feedback_text(chord_result)

            # 在画面上方绘制评价文字
            text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (output_frame.shape[1] - text_size[0]) // 2
            cv2.putText(output_frame, feedback_text, (text_x, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 显示当前事件信息
            ev_info = f"{self.song.current_index+1}/{len(self.song.events)}"
            if self.song.type == 'chord':
                ev_name = self.song.events[self.song.current_index]['chord']
            else:
                ev = self.song.events[self.song.current_index]
                ev_name = f"String {ev['string']} Fret {ev['fret']}"
            cv2.putText(output_frame, f"{self.song.title} - {ev_info}",
                        (output_frame.shape[1] - 400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(output_frame, f"Now: {ev_name}",
                        (output_frame.shape[1] - 400, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 绘制指板图（如果开启）
            if self.show_diagram:
                output_frame = draw_fretboard_diagram(output_frame, self.current_target, alpha=0.5)

            # 统计正确率
            total = sum(1 for s, f in self.current_target.items() if f != 0)
            correct = sum(1 for s, status in chord_result.items() if status == 'correct')
            if total > 0:
                stats_text += f" | 正确率: {correct}/{total} ({correct*100//total}%)"

        else:
            if not self.song:
                feedback_text = "请先加载歌曲"
            else:
                feedback_text = "未检测到手"

        # 添加基础信息
        output_frame = self.display.add_info(output_frame, 30, len(results.hand_landmarks) if results.hand_landmarks else 0)

        # 转换为 RGB
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        return output_frame_rgb, stats_text, feedback_text

# ---------- 创建 Gradio 界面 ----------
def create_ui():
    app = iGuitarWebUI()

    with gr.Blocks(title="🎸 iGuitar WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎸 iGuitar - 吉他学习辅助系统")

        with gr.Row():
            with gr.Column(scale=2):
                # 视频区域
                video = gr.Image(label="摄像头画面", height=540)
                # 关键反馈放在视频下方
                with gr.Row():
                    stats = gr.Textbox(label="实时统计", interactive=False, scale=1)
                    feedback = gr.Textbox(label="错误反馈", interactive=False, scale=1)
            with gr.Column(scale=1):
                # 左侧控制面板，使用折叠面板精简
                with gr.Accordion("系统控制", open=True):
                    init_btn = gr.Button("🚀 初始化系统", variant="primary")
                    status = gr.Textbox(label="状态", interactive=False)
                    info = gr.Textbox(label="信息", interactive=False)

                with gr.Accordion("歌曲选择", open=True):
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                        song_dropdown = gr.Dropdown(
                            choices=[],
                            label="选择歌曲",
                            interactive=True
                        )
                    load_btn = gr.Button("📂 加载歌曲", variant="secondary")
                    song_status = gr.Textbox(label="歌曲状态", interactive=False)

                with gr.Accordion("显示控制", open=False):
                    with gr.Row():
                        target_btn = gr.Button("🎯 切换目标点")
                        diagram_btn = gr.Button("🎸 切换指板图")
                    target_status = gr.Textbox(label="目标点状态", interactive=False)
                    diagram_status = gr.Textbox(label="指板图状态", interactive=False)

                with gr.Accordion("练习控制", open=True):
                    with gr.Row():
                        prev_btn = gr.Button("⬅️ 上一个")
                        next_btn = gr.Button("➡️ 下一个")
                    chord_info = gr.Textbox(label="当前事件", interactive=False)

        # 初始化回调
        init_btn.click(app.initialize, outputs=[status, info])
        refresh_btn.click(app.refresh_songs, outputs=[song_dropdown])
        load_btn.click(app.load_song, inputs=[song_dropdown], outputs=[song_status, chord_info])

        # 控制按钮回调
        next_btn.click(app.next_chord, outputs=[chord_info])
        prev_btn.click(app.prev_chord, outputs=[chord_info])
        target_btn.click(app.toggle_target, outputs=[target_status])
        diagram_btn.click(app.toggle_diagram, outputs=[diagram_status])

        # 定时更新画面
        timer = gr.Timer(0.03)
        timer.tick(app.get_frame, outputs=[video, stats, feedback])

        # 初始刷新歌曲列表
        demo.load(fn=app.refresh_songs, outputs=[song_dropdown])

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)