#!/usr/bin/env python3
# scripts/main.py
import time
import os
import sys
import cv2
import numpy as np
import json   # 新增导入

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core import Camera, HandTracker, VideoPreprocessor
from src.ui.display import Display
from src.ui.drawing import draw_target_points, draw_missing_markers
from src.mapping import FretboardMapper
from src.data import Song

def draw_fretboard_diagram(frame, target, position='top-right', size=(200, 300), alpha=0.5):
    """
    在frame上绘制一个简化的指板图，显示目标按弦位置。
    弦顺序：6弦（最粗）在上方，1弦（最细）在下方。
    """
    h, w = frame.shape[:2]
    diagram_w, diagram_h = size

    # 确定指板图的位置
    if position == 'top-right':
        x_start = w - diagram_w - 10
        y_start = 10
    else:
        x_start = w - diagram_w - 10
        y_start = h - diagram_h - 10

    # 创建指板图背景（半透明）
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + diagram_w, y_start + diagram_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 绘制弦（6根，从上到下依次为6弦、5弦...1弦）
    string_spacing = diagram_h / 7  # 留出边距，从 y_start+20 开始
    for i in range(6):
        y = int(y_start + 20 + i * string_spacing)
        cv2.line(frame, (x_start + 20, y), (x_start + diagram_w - 20, y), (255, 255, 255), 1)

    # 绘制品丝（12品，简化为每隔一定距离画竖线）
    fret_spacing = (diagram_w - 40) / 12
    for i in range(13):
        x = int(x_start + 20 + i * fret_spacing)
        cv2.line(frame, (x, y_start + 20), (x, y_start + diagram_h - 20), (255, 255, 255), 1)

    # 绘制目标点
    for string, fret in target.items():
        if fret == 0:
            continue
        # 弦号1-6，6弦在顶部（索引0），1弦在底部（索引5）
        string_idx = 6 - string
        y = int(y_start + 20 + string_idx * string_spacing + string_spacing / 2)
        x = int(x_start + 20 + (fret - 1) * fret_spacing + fret_spacing / 2)
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)

    # 标注弦号（左侧显示，从上到下依次为6,5,4,3,2,1）
    for i in range(6):
        y = int(y_start + 20 + i * string_spacing + string_spacing / 2)
        cv2.putText(frame, str(6 - i), (x_start + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame

def draw_mini_fretboard_diagram(frame, target, x_offset, y_offset, size=(180, 100), alpha=0.7):
    """
    迷你模式专用的指板图，绘制在指定区域。
    """
    diagram_w, diagram_h = size
    x_start = x_offset
    y_start = y_offset

    # 背景半透明
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + diagram_w, y_start + diagram_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 弦
    string_spacing = diagram_h / 7
    for i in range(6):
        y = int(y_start + 20 + i * string_spacing)
        cv2.line(frame, (x_start + 20, y), (x_start + diagram_w - 20, y), (255, 255, 255), 1)

    # 品丝（只画5根示意）
    fret_spacing = (diagram_w - 40) / 12
    for i in range(0, 13, 3):
        x = int(x_start + 20 + i * fret_spacing)
        cv2.line(frame, (x, y_start + 20), (x, y_start + diagram_h - 20), (255, 255, 255), 1)

    # 目标点
    for string, fret in target.items():
        if fret == 0:
            continue
        string_idx = 6 - string
        y = int(y_start + 20 + string_idx * string_spacing + string_spacing / 2)
        x = int(x_start + 20 + (fret - 1) * fret_spacing + fret_spacing / 2)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

    return frame

def get_feedback_text(chord_result):
    """生成错误评价文字（英文）"""
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

def check_model_file():
    """检查模型文件是否存在"""
    model_path = 'assets/models/hand_landmarker.task'
    if not os.path.exists(model_path):
        print("\n❌ Model file not found!")
        print(f"Please place hand_landmarker.task in {model_path}")
        print("\nDownload link:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return False
    return True

def main():
    print("=" * 50)
    print("🎸 iGuitar - Guitar Learning Assistant v0.6 (with Song Menu)")
    print("=" * 50)

    if not check_model_file():
        return

    if not os.path.exists('calibration_matrix.npy'):
        print("\n⚠️ Calibration matrix file 'calibration_matrix.npy' not found.")
        print("Please run calibrate.py first.")
        return

    # 初始化模块
    try:
        camera = Camera(camera_id=0)
        preprocessor = VideoPreprocessor(flip=True, target_width=1280)
        hand_tracker = HandTracker(max_hands=2)
        display = Display()
        mapper = FretboardMapper('calibration_matrix.npy')
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 曲目选择菜单 ==========
    songs_dir = 'assets/songs'
    if not os.path.exists(songs_dir):
        print(f"⚠️ Song directory not found: {songs_dir}")
        return

    # 扫描所有 .json 文件
    song_files = [f for f in os.listdir(songs_dir) if f.endswith('.json')]
    if not song_files:
        print(f"⚠️ No .json song files found in {songs_dir}")
        return

    # 打印菜单
    print("\n" + "=" * 50)
    print("Available Songs:")
    song_titles = []
    for i, filename in enumerate(song_files, 1):
        # 尝试读取曲目标题
        try:
            with open(os.path.join(songs_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                title = data.get('title', filename)
        except:
            title = filename
        song_titles.append(title)
        print(f"{i}. {title} ({filename})")
    print("0. Exit")
    print("=" * 50)

    # 获取用户选择
    while True:
        try:
            choice = input("Enter the number of the song to load: ").strip()
            if choice == '0':
                print("Exiting.")
                return
            idx = int(choice) - 1
            if 0 <= idx < len(song_files):
                selected_file = song_files[idx]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(song_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    song_file = os.path.join(songs_dir, selected_file)
    try:
        song = Song(song_file)
        current_target = song.get_current_target()
        print(f"✅ Loaded song: {song.title}, {len(song.events)} events")
    except Exception as e:
        print(f"❌ Failed to load song: {e}")
        return

    # ========== 主程序逻辑 ==========
    print("\n💡 Place your left hand in front of the camera.")
    print("💡 Controls: N-next, P-prev, R-reset, T-toggle target points, M-mini mode, D-toggle diagram, Q-quit")
    print("💡 Colors: Green-correct, Red-wrong fret, Orange-extra finger, White-unknown")
    print("💡 Lines: White lines show deviation from target points\n")

    # 状态变量
    show_target = True
    mini_mode = False
    show_diagram = True
    prev_time = time.time()
    frame_count = 0

    # 指尖关键点索引映射
    finger_indices = {
        'thumb': 4,
        'index': 8,
        'middle': 12,
        'ring': 16,
        'little': 20
    }

    # 存储当前事件信息
    current_ev_name = ""
    current_feedback = ""
    current_ev_info = ""

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                print("❌ Failed to read camera")
                break

            rgb_frame, bgr_frame = preprocessor.process(frame)
            results = hand_tracker.detect(rgb_frame)
            output_frame = display.draw_landmarks(bgr_frame, results)

            fingertip_pixels = {}

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                actual_fingers = mapper.get_finger_frets(hand_landmarks, output_frame.shape)

                # 获取指尖像素坐标
                for finger, idx in finger_indices.items():
                    lm = hand_landmarks[idx]
                    h, w, _ = output_frame.shape
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    fingertip_pixels[finger] = (x, y)

                # 比对
                chord_result = {}
                for string in range(1, 7):
                    chord_result[string] = 'unknown'

                for finger, pos in actual_fingers.items():
                    if pos is None:
                        continue
                    fret, string = pos
                    expected_fret = current_target.get(string, 0)
                    if expected_fret == fret:
                        chord_result[string] = 'correct'
                    else:
                        if expected_fret != 0:
                            chord_result[string] = 'wrong'
                        else:
                            chord_result[string] = 'extra'

                for string, expected_fret in current_target.items():
                    if expected_fret != 0:
                        found = any(pos is not None and pos[1] == string for pos in actual_fingers.values())
                        if not found:
                            chord_result[string] = 'missing'

                # 生成评价文字和事件信息
                current_feedback = get_feedback_text(chord_result)
                ev_info = f"{song.current_index+1}/{len(song.events)}"
                if song.type == 'chord':
                    ev_name = song.events[song.current_index]['chord']
                else:
                    ev = song.events[song.current_index]
                    ev_name = f"String {ev['string']} Fret {ev['fret']}"
                current_ev_name = ev_name
                current_ev_info = ev_info

                # 正常模式下的完整绘制
                if not mini_mode:
                    if show_target:
                        output_frame = draw_target_points(output_frame, mapper, current_target)
                    output_frame = draw_missing_markers(output_frame, mapper, current_target, actual_fingers)

                    # 指尖圆点和连线
                    for finger, pos in actual_fingers.items():
                        if pos is None:
                            continue
                        fret, string = pos
                        expected_fret = current_target.get(string, 0)
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
                            target_x, target_y = mapper.fretboard_to_pixel(expected_fret, string)
                            cv2.line(output_frame, (x, y), (target_x, target_y), (255, 255, 255), 2)

                    # 文字评价
                    text_size = cv2.getTextSize(current_feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = (output_frame.shape[1] - text_size[0]) // 2
                    cv2.putText(output_frame, current_feedback, (text_x, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # 事件信息
                    cv2.putText(output_frame, f"{song.title} - {ev_info}",
                                (output_frame.shape[1] - 400, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(output_frame, f"Now: {ev_name}",
                                (output_frame.shape[1] - 400, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if show_diagram:
                        output_frame = draw_fretboard_diagram(output_frame, current_target, alpha=0.5)

            else:
                current_feedback = "No hand detected"
                if not mini_mode:
                    cv2.putText(output_frame, "No hand detected", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
            if not mini_mode:
                output_frame = display.add_info(output_frame, fps, hand_count)

            # 显示窗口
            if mini_mode:
                mini_frame = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(mini_frame, f"{song.title}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(mini_frame, f"Event: {ev_info}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(mini_frame, f"Now: {current_ev_name}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                feedback_lines = current_feedback.split(" | ")
                y_pos = 120
                for line in feedback_lines[:2]:
                    cv2.putText(mini_frame, line, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if "wrong" in line or "extra" in line or "missing" in line else (0, 255, 0), 1)
                    y_pos += 20

                if show_diagram:
                    dia_w, dia_h = 180, 100
                    x_offset = mini_frame.shape[1] - dia_w - 10
                    y_offset = mini_frame.shape[0] - dia_h - 10
                    mini_frame = draw_mini_fretboard_diagram(mini_frame, current_target, x_offset, y_offset, size=(dia_w, dia_h), alpha=0.7)

                cv2.imshow("iGuitar Mini", mini_frame)
                try:
                    cv2.destroyWindow('iGuitar - Visual Feedback')
                except:
                    pass
            else:
                cv2.imshow('iGuitar - Visual Feedback', output_frame)
                try:
                    cv2.destroyWindow('iGuitar Mini')
                except:
                    pass

            # 按键
            key = display.wait_key(1)
            if key == ord('q'):
                print("\n👋 User quit")
                break
            elif key == ord('n'):
                if song.next():
                    current_target = song.get_current_target()
                    print(f"➡️ Event {song.current_index+1}/{len(song.events)}")
            elif key == ord('p'):
                if song.prev():
                    current_target = song.get_current_target()
                    print(f"⬅️ Event {song.current_index+1}/{len(song.events)}")
            elif key == ord('r'):
                song.reset()
                current_target = song.get_current_target()
                print("🔄 Song reset")
            elif key == ord('t'):
                show_target = not show_target
                print(f"Show target points: {show_target}")
            elif key == ord('m'):
                mini_mode = not mini_mode
                print(f"Mini mode: {mini_mode}")
            elif key == ord('d'):
                show_diagram = not show_diagram
                print(f"Show fretboard diagram: {show_diagram}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        hand_tracker.close()
        display.destroy_all()
        print(f"\n✅ Program exited safely. Processed {frame_count} frames")

if __name__ == "__main__":
    main()