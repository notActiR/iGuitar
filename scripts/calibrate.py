# scripts/calibrate.py
import cv2
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.core.preprocessor import VideoPreprocessor
from src.ui.display import Display
from src.mapping.calibrator import Calibrator

def main():
    camera = Camera(camera_id=0)
    # 标定分辨率与主程序保持一致（640）
    preprocessor = VideoPreprocessor(flip=True, target_width=640)
    hand_tracker = HandTracker(max_hands=1)
    display = Display()
    calibrator = Calibrator()

    print("=" * 50)
    print("🎸 iGuitar 9-Point Calibration Tool (640px)")
    print("Please use your index finger to touch the following points:")
    points_info = [
        (1, 1, "1st string, 1st fret"),
        (1, 6, "1st string, 6th fret"),
        (1, 12, "1st string, 12th fret"),
        (3, 1, "3rd string, 1st fret"),
        (3, 6, "3rd string, 6th fret"),
        (3, 12, "3rd string, 12th fret"),
        (6, 1, "6th string, 1st fret"),
        (6, 6, "6th string, 6th fret"),
        (6, 12, "6th string, 12th fret")
    ]
    step = 0

    while step < len(points_info):
        ret, frame = camera.read_frame()
        if not ret:
            break

        rgb_frame, bgr_frame = preprocessor.process(frame)
        results = hand_tracker.detect(rgb_frame)

        # 只显示食指指尖
        if results.hand_landmarks:
            hand_lms = results.hand_landmarks[0]
            index_tip = hand_lms[8]
            h, w = bgr_frame.shape[:2]
            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)
            cv2.circle(bgr_frame, (cx, cy), 15, (0, 255, 0), -1)
            cv2.circle(bgr_frame, (cx, cy), 15, (0, 0, 255), 3)
            cv2.putText(bgr_frame, "> Use this fingertip <", (cx+20, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(bgr_frame, "No hand detected", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        fret, string, desc = points_info[step]
        cv2.putText(bgr_frame, f"Step {step+1}/{len(points_info)}: Touch {desc}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bgr_frame, "Press SPACE to record", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(bgr_frame, "Press Q to quit", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        display.show("Calibration", bgr_frame)
        key = display.wait_key(1)

        if key == ord(' '):
            if results.hand_landmarks:
                lm = results.hand_landmarks[0][8]
                h, w = bgr_frame.shape[:2]
                x = int(lm.x * w)
                y = int(lm.y * h)
                calibrator.add_point(x, y, fret, string)
                print(f"Recorded {desc}: pixel({x},{y}) -> fretboard({fret},{string})")
                step += 1
            else:
                print("No hand detected")
        elif key == ord('q'):
            break

    if step == len(points_info):
        H = calibrator.compute_homography()
        calibrator.save('calibration_matrix.npy')
        print("Calibration completed! Matrix saved as calibration_matrix.npy")
    else:
        print("Calibration incomplete")

    camera.release()
    hand_tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()