"""
标定脚本：运行此脚本，按屏幕提示用食指依次点按四个点，生成标定矩阵
"""
import sys
import os
import cv2

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.core import Camera, HandTracker, VideoPreprocessor
from src.ui.display import Display
from src.mapping import Calibrator

def main():
    # 初始化模块
    camera = Camera()
    preprocessor = VideoPreprocessor(flip=True)
    hand_tracker = HandTracker()
    display = Display()
    calibrator = Calibrator()

    print("=" * 50)
    print("🎸 iGuitar Calibration Tool")
    print("Please use your index finger to touch the following four points:")
    print("1. 1st string, 1st fret")
    print("2. 6th string, 1st fret")
    print("3. 1st string, 12th fret")
    print("4. 6th string, 12th fret")
    print("Press SPACE to record the current index fingertip position")
    print("Press Q to quit")
    print("=" * 50)

    # 定义四个点 (品, 弦, 描述)
    points_info = [
        (1, 1, "1st string, 1st fret"),
        (6, 1, "6th string, 1st fret"),
        (1, 12, "1st string, 12th fret"),
        (6, 12, "6th string, 12th fret")
    ]
    step = 0

    while step < 4:
        ret, frame = camera.read_frame()
        if not ret:
            print("Failed to read camera")
            break

        rgb_frame, bgr_frame = preprocessor.process(frame)
        results = hand_tracker.detect(rgb_frame)

        # 绘制手部骨架
        frame = display.draw_landmarks(bgr_frame, results)

        # 显示当前步骤提示（英文）
        fret, string, desc = points_info[step]
        cv2.putText(frame, f"Step {step+1}: Touch {desc}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to record", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Press Q to quit", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        display.show('Calibration', frame)
        key = display.wait_key(1)

        if key == ord(' '):
            # 获取食指指尖 (landmark 8)
            if results.hand_landmarks:
                # 取第一只手
                hand_landmarks = results.hand_landmarks[0]
                lm = hand_landmarks[8]  # 食指指尖
                h, w, _ = frame.shape
                x = int(lm.x * w)
                y = int(lm.y * h)
                calibrator.add_point(x, y, fret, string)
                print(f"✅ Recorded {desc}: pixel({x}, {y}) -> fretboard({fret}, {string})")
                step += 1
            else:
                print("⚠️ No hand detected, please keep your hand in frame")
        elif key == ord('q'):
            print("User quit calibration")
            break

    if step == 4:
        # 计算并保存矩阵
        H = calibrator.compute_homography()
        calibrator.save('calibration_matrix.npy')
        print("🎉 Calibration complete! Matrix saved as calibration_matrix.npy")
    else:
        print("⚠️ Calibration incomplete, matrix not saved")

    # 清理资源
    camera.release()
    hand_tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()