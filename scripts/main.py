#!/usr/bin/env python3
# scripts/main.py (UI优化版)
import time
import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core import Camera, HandTracker, VideoPreprocessor
from src.ui.display import Display
from src.mapping import FretboardMapper
from src.data import Song
from src.data.chord_db import CHORDS

# ---------- 辅助绘图函数 ----------
def draw_target_points(frame, mapper, target, alpha=0.2):
    """绘制半透明绿色目标点（预期位置）"""
    overlay = frame.copy()
    for string, fret in target.items():
        if fret == 0:
            continue
        x, y = mapper.fretboard_to_pixel(fret, string)
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
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
        cv2.circle(frame, (x, y), 12, (128, 128, 128), 2)
    return frame

def draw_fretboard_diagram(frame, target, position='bottom-right', size=(160, 200), alpha=0.5):
    """
    绘制简化的指板图（右下角，更小）
    """
    h, w = frame.shape[:2]
    diagram_w, diagram_h = size
    if position == 'bottom-right':
        x_start = w - diagram_w - 10
        y_start = h - diagram_h - 10
    else:
        x_start = w - diagram_w - 10
        y_start = 10

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

    # 品丝编号（仅显示 1, 3, 5, 7, 9, 12）
    for i in [1, 3, 5, 7, 9, 12]:
        if i <= 12:
            x = int(x_start + 20 + (i-1) * fret_spacing + fret_spacing/2)
            cv2.putText(frame, str(i), (x-5, y_start+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    for string, fret in target.items():
        if fret == 0:
            continue
        string_idx = 6 - string
        y = int(y_start + 20 + string_idx * string_spacing + string_spacing / 2)
        x = int(x_start + 20 + (fret - 1) * fret_spacing + fret_spacing / 2)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

    # 弦号标注（仅标注 6 和 1）
    for i in [0, 5]:
        y = int(y_start + 20 + i * string_spacing + string_spacing / 2)
        cv2.putText(frame, str(6 - i), (x_start + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    return frame

def draw_mini_fretboard_diagram(frame, target, x_offset, y_offset, size=(140, 80), alpha=0.7):
    """迷你模式专用的指板图（更小）"""
    diagram_w, diagram_h = size
    x_start = x_offset
    y_start = y_offset

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + diagram_w, y_start + diagram_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    string_spacing = diagram_h / 7
    for i in range(6):
        y = int(y_start + 20 + i * string_spacing)
        cv2.line(frame, (x_start + 20, y), (x_start + diagram_w - 20, y), (255, 255, 255), 1)

    fret_spacing = (diagram_w - 40) / 12
    for i in range(0, 13, 3):
        x = int(x_start + 20 + i * fret_spacing)
        cv2.line(frame, (x, y_start + 20), (x, y_start + diagram_h - 20), (255, 255, 255), 1)

    for string, fret in target.items():
        if fret == 0:
            continue
        string_idx = 6 - string
        y = int(y_start + 20 + string_idx * string_spacing + string_spacing / 2)
        x = int(x_start + 20 + (fret - 1) * fret_spacing + fret_spacing / 2)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
    return frame

def get_feedback_text(chord_result):
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
    model_path = 'assets/models/hand_landmarker.task'
    if not os.path.exists(model_path):
        print("\n❌ Model file not found!")
        print(f"Please place hand_landmarker.task in {model_path}")
        print("\nDownload link:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return False
    return True

def get_event_name(song):
    if not song or song.current_index >= len(song.events):
        return ""
    ev = song.events[song.current_index]
    if song.type == 'chord':
        return ev['chord']
    elif song.type == 'note':
        return f"String {ev['string']} Fret {ev['fret']}"
    elif song.type == 'custom':
        target = ev['target']
        notes = [f"{s}:{f}" for s, f in target.items() if f != 0]
        return " ".join(notes) if notes else "No notes"
    else:
        return "Unknown"

def main():
    print("=" * 50)
    print("🎸 iGuitar - Guitar Learning Assistant (UI Optimized)")
    print("=" * 50)

    if not check_model_file():
        return
    if not os.path.exists('calibration_matrix.npy'):
        print("\n⚠️ Calibration matrix not found. Run calibrate.py first.")
        return

    try:
        camera = Camera(camera_id=0)
        preprocessor = VideoPreprocessor(flip=True, target_width=640)
        hand_tracker = HandTracker(max_hands=2, use_enhancer=True)   # 启用论文增强
        display = Display()
        mapper = FretboardMapper('calibration_matrix.npy')
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    # 尝试加载默认歌曲
    default_song = 'assets/songs/twinkle.json'
    if os.path.exists(default_song):
        try:
            song = Song(default_song)
            current_target = song.get_current_target()
            print(f"✅ Loaded default song: {song.title}, {len(song.events)} events")
        except Exception as e:
            print(f"⚠️ Failed to load default song: {e}")
            song = None
            current_target = None
    else:
        song = None
        current_target = None
        print("ℹ️ No default song found. Press L to load a file.")

    print("\nControls:")
    print("  N/P - prev/next   R - reset   T - toggle targets   M - mini mode")
    print("  D - toggle diagram   SPACE - align   A - auto advance   L - load file   Q - quit\n")

    align_mode = False
    auto_advance = True
    stable_counter = 0
    ALIGN_FRAMES = 15
    ADVANCE_FRAMES = 30

    show_target = True
    mini_mode = False
    show_diagram = True
    prev_time = time.time()
    frame_count = 0

    finger_indices = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'little':20}
    current_feedback = ""
    current_ev_info = ""

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            rgb_frame, bgr_frame = preprocessor.process(frame)
            results = hand_tracker.detect(rgb_frame)
            output_frame = display.draw_landmarks(bgr_frame, results)

            if results.hand_landmarks and song and current_target:
                hand_landmarks = results.hand_landmarks[0]
                actual_fingers = mapper.get_finger_frets(hand_landmarks, output_frame.shape)

                fingertip_pixels = {}
                for finger, idx in finger_indices.items():
                    lm = hand_landmarks[idx]
                    h, w = output_frame.shape[:2]
                    x, y = int(lm.x * w), int(lm.y * h)
                    fingertip_pixels[finger] = (x, y)

                chord_result = {s: 'unknown' for s in range(1,7)}
                for finger, pos in actual_fingers.items():
                    if pos is None: continue
                    fret, string = pos
                    expected_fret = current_target.get(string, 0)
                    if expected_fret == fret:
                        chord_result[string] = 'correct'
                    else:
                        chord_result[string] = 'wrong' if expected_fret != 0 else 'extra'
                for string, expected_fret in current_target.items():
                    if expected_fret != 0:
                        if not any(pos is not None and pos[1] == string for pos in actual_fingers.values()):
                            chord_result[string] = 'missing'

                current_feedback = get_feedback_text(chord_result)
                ev_info = f"{song.current_index+1}/{len(song.events)}"
                ev_name = get_event_name(song)

                all_correct = all(chord_result.get(s) == 'correct' for s in current_target.keys() if current_target[s] != 0)
                no_extra = all(chord_result.get(s) != 'extra' for s in range(1,7))
                is_perfect = all_correct and no_extra

                if align_mode:
                    if is_perfect:
                        stable_counter += 1
                        if stable_counter >= ALIGN_FRAMES:
                            align_mode = False
                            stable_counter = 0
                            print("✅ Aligned!")
                    else:
                        stable_counter = 0
                else:
                    if auto_advance and is_perfect:
                        stable_counter += 1
                        if stable_counter >= ADVANCE_FRAMES:
                            if song.next():
                                current_target = song.get_current_target()
                                print(f"⏩ Auto advance to {song.current_index+1}")
                            stable_counter = 0
                    else:
                        stable_counter = max(0, stable_counter - 2)

                if not mini_mode:
                    if show_target:
                        output_frame = draw_target_points(output_frame, mapper, current_target, alpha=0.2)
                    output_frame = draw_missing_markers(output_frame, mapper, current_target, actual_fingers)

                    for finger, pos in actual_fingers.items():
                        if pos is None: continue
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
                        cv2.circle(output_frame, (x, y), 8, color, -1)
                        cv2.circle(output_frame, (x, y), 8, (255, 255, 255), 2)
                        cv2.putText(output_frame, f"{fret},{string}", (x+15, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        if expected_fret != 0:
                            tx, ty = mapper.fretboard_to_pixel(expected_fret, string)
                            cv2.line(output_frame, (x, y), (tx, ty), (255, 255, 200), 1)

                    # 评价文字移到画面底部中央
                    feedback = current_feedback
                    text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = (output_frame.shape[1] - text_size[0]) // 2
                    text_y = output_frame.shape[0] - 30
                    cv2.rectangle(output_frame, (text_x-10, text_y-25), (text_x+text_size[0]+10, text_y+5), (0,0,0), -1)
                    cv2.putText(output_frame, feedback, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                    # 曲目信息（右上角）
                    cv2.putText(output_frame, f"{song.title} - {ev_info}", (output_frame.shape[1]-300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    cv2.putText(output_frame, f"Now: {ev_name}", (output_frame.shape[1]-300, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                    if show_diagram:
                        output_frame = draw_fretboard_diagram(output_frame, current_target, position='bottom-right', alpha=0.5)

                    if align_mode:
                        cv2.putText(output_frame, "ALIGN MODE: Play current chord", (50,150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    elif auto_advance:
                        bar_width = int((stable_counter / ADVANCE_FRAMES) * 200)
                        cv2.rectangle(output_frame, (50, 170), (250, 185), (0,100,0), -1)
                        cv2.rectangle(output_frame, (50, 170), (50+bar_width, 185), (0,200,0), -1)
                        cv2.putText(output_frame, "Auto advance", (50, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    else:
                        cv2.putText(output_frame, "Auto advance OFF (N/P)", (50,150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            elif song is None:
                cv2.putText(output_frame, "No song. Press L to load file.", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.putText(output_frame, "No hand detected", (10,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # FPS 和手数信息
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
            if not mini_mode:
                output_frame = display.add_info(output_frame, fps, hand_count)

            # 迷你模式（窗口改为 180x360）
            if mini_mode:
                mini_frame = np.zeros((180, 360, 3), dtype=np.uint8)
                title = song.title if song else "No Song"
                ev_info_str = ev_info if song else ""
                ev_name_str = ev_name if song else ""
                cv2.putText(mini_frame, f"{title}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                cv2.putText(mini_frame, f"Event: {ev_info_str}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                cv2.putText(mini_frame, f"Now: {ev_name_str}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                lines = current_feedback.split(" | ")
                y = 120
                for line in lines[:2]:
                    cv2.putText(mini_frame, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255) if "wrong" in line or "extra" in line or "missing" in line else (0,255,0), 1)
                    y += 18
                if show_diagram and song and current_target:
                    mini_frame = draw_mini_fretboard_diagram(mini_frame, current_target,
                                                             mini_frame.shape[1]-150, mini_frame.shape[0]-90,
                                                             size=(140,80), alpha=0.7)
                cv2.imshow("iGuitar Mini", mini_frame)
                try:
                    cv2.destroyWindow('iGuitar - Visual Feedback')
                except: pass
            else:
                cv2.imshow('iGuitar - Visual Feedback', output_frame)
                try:
                    cv2.destroyWindow('iGuitar Mini')
                except: pass

            key = display.wait_key(1)
            if key == ord('q'):
                break
            elif key == ord('n') and song:
                if song.next():
                    current_target = song.get_current_target()
                    print(f"➡️ Event {song.current_index+1}")
                    align_mode = False
                    stable_counter = 0
            elif key == ord('p') and song:
                if song.prev():
                    current_target = song.get_current_target()
                    print(f"⬅️ Event {song.current_index+1}")
                    align_mode = False
                    stable_counter = 0
            elif key == ord('r') and song:
                song.reset()
                current_target = song.get_current_target()
                align_mode = False
                stable_counter = 0
                print("🔄 Reset")
            elif key == ord('t'):
                show_target = not show_target
                print(f"Target points: {show_target}")
            elif key == ord('m'):
                mini_mode = not mini_mode
                print(f"Mini mode: {mini_mode}")
            elif key == ord('d'):
                show_diagram = not show_diagram
                print(f"Diagram: {show_diagram}")
            elif key == ord(' '):
                if song:
                    align_mode = True
                    stable_counter = 0
                    print("🔍 Align mode ON. Play the current chord/note correctly.")
                else:
                    print("No song loaded. Press L to load.")
            elif key == ord('a'):
                auto_advance = not auto_advance
                print(f"Auto advance: {'ON' if auto_advance else 'OFF'}")
                if not auto_advance:
                    stable_counter = 0
            elif key == ord('l'):
                cv2.destroyAllWindows()
                print("\n🎸 Select file (JSON or GP3/GP4/GP5)")
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    filepath = filedialog.askopenfilename(
                        title="Select Song File",
                        filetypes=[
                            ("JSON曲目", "*.json"),
                            ("Guitar Pro files", "*.gp3 *.gp4 *.gp5"),
                            ("All files", "*.*")
                        ]
                    )
                    root.destroy()
                except ImportError:
                    filepath = input("> ").strip()
                if filepath and os.path.exists(filepath):
                    try:
                        if filepath.lower().endswith('.json'):
                            new_song = Song(filepath)
                        else:
                            new_song = Song.from_gp_file(filepath)
                        song = new_song
                        current_target = song.get_current_target()
                        print(f"✅ Loaded: {song.title}, {len(song.events)} events (type={song.type})")
                        align_mode = False
                        stable_counter = 0
                    except Exception as e:
                        print(f"❌ Failed: {e}")
                else:
                    print("No file selected.")
            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        hand_tracker.close()
        display.destroy_all()
        print(f"Exited, processed {frame_count} frames")

if __name__ == "__main__":
    main()