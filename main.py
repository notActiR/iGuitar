import time
import os
import cv2
import numpy as np
from camera.camera import Camera
from vision.preprocess import VideoPreprocessor
from vision.hand_tracker import HandTracker
from ui.display import Display
from mapping.fretboard_mapper import FretboardMapper
from data.song import Song

def check_model_file():
    """检查模型文件是否存在"""
    model_path = 'models/hand_landmarker.task'
    if not os.path.exists(model_path):
        print("\n❌ Model file not found!")
        print(f"Please place hand_landmarker.task in {model_path}")
        print("\nDownload link:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return False
    return True

def main():
    print("=" * 50)
    print("🎸 iGuitar - Guitar Learning Assistant v0.3 (Song Practice Mode)")
    print("=" * 50)

    # Check model file
    if not check_model_file():
        return

    # Check calibration matrix
    if not os.path.exists('calibration_matrix.npy'):
        print("\n⚠️ Calibration matrix file 'calibration_matrix.npy' not found.")
        print("Please run calibrate.py first.")
        return

    # Initialize modules
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

    # Load song (modify path as needed)
    song_file = 'songs/twinkle.json'   # Make sure this file exists and has English title
    try:
        song = Song(song_file)
        current_target = song.get_current_target()
        print(f"✅ Loaded song: {song.title}, {len(song.events)} events")
    except Exception as e:
        print(f"❌ Failed to load song: {e}")
        return

    print("\n💡 Place your left hand in front of the camera.")
    print("💡 Controls: N-next event, P-previous event, R-reset song, Q-quit")
    print("💡 Finger colors: Green-correct, Red-wrong fret, Orange-extra finger, White-unknown\n")

    # FPS calculation
    prev_time = time.time()
    frame_count = 0

    # Finger landmark indices
    finger_indices = {
        'thumb': 4,
        'index': 8,
        'middle': 12,
        'ring': 16,
        'little': 20
    }

    try:
        while True:
            # 1. Read frame
            ret, frame = camera.read_frame()
            if not ret:
                print("❌ Failed to read camera frame")
                break

            # 2. Preprocess
            rgb_frame, bgr_frame = preprocessor.process(frame)

            # 3. Hand detection
            results = hand_tracker.detect(rgb_frame)

            # 4. Draw skeleton
            output_frame = display.draw_landmarks(bgr_frame, results)

            # 5. Finger analysis and feedback
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                actual_fingers = mapper.get_finger_frets(hand_landmarks, output_frame.shape)

                # Compare: generate status for each string
                chord_result = {}
                for string in range(1, 7):
                    chord_result[string] = 'unknown'

                # Process fingers that are on fretboard
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

                # Mark missing strings (expected but no finger)
                for string, expected_fret in current_target.items():
                    if expected_fret != 0:
                        found = any(pos is not None and pos[1] == string for pos in actual_fingers.values())
                        if not found:
                            chord_result[string] = 'missing'

                # Draw fingertip dots
                for finger, pos in actual_fingers.items():
                    if pos is None:
                        continue
                    fret, string = pos
                    idx = finger_indices[finger]
                    lm = hand_landmarks[idx]
                    h, w, _ = output_frame.shape
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    status = chord_result.get(string, 'unknown')
                    if status == 'correct':
                        color = (0, 255, 0)
                    elif status == 'wrong':
                        color = (0, 0, 255)
                    elif status == 'extra':
                        color = (0, 165, 255)   # orange
                    else:
                        color = (255, 255, 255) # white

                    cv2.circle(output_frame, (x, y), 10, color, -1)
                    cv2.circle(output_frame, (x, y), 10, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"{fret},{string}", (x+15, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Display current event info (all in English)
                ev_info = f"{song.current_index+1}/{len(song.events)}"
                if song.type == 'chord':
                    ev_name = song.events[song.current_index]['chord']
                else:
                    ev = song.events[song.current_index]
                    ev_name = f"String {ev['string']} Fret {ev['fret']}"

                # Top-right corner info
                cv2.putText(output_frame, f"{song.title} - {ev_info}",
                            (output_frame.shape[1] - 400, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(output_frame, f"Now: {ev_name}",
                            (output_frame.shape[1] - 400, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            else:
                cv2.putText(output_frame, "No hand detected", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # 6. Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # 7. Add info (FPS, hand count)
            hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
            output_frame = display.add_info(output_frame, fps, hand_count)

            # 8. Show frame
            display.show('iGuitar - Song Practice', output_frame)

            # 9. Key handling
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