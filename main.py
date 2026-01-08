import time
import os
from camera.camera import Camera
from vision.preprocess import VideoPreprocessor
from vision.hand_tracker import HandTracker
from ui.display import Display


def check_model_file():
    """检查模型文件是否存在"""
    model_path = 'models/hand_landmarker.task'
    if not os.path.exists(model_path):
        print("\n❌ 找不到模型文件！")
        print(f"请将 hand_landmarker.task 放到 {model_path}")
        print("\n下载地址:")
        print(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return False
    return True


def main():
    print("=" * 50)
    print("🎸 iGuitar - 吉他学习辅助系统 v0.1")
    print("   使用新版 MediaPipe Tasks API")
    print("=" * 50)

    # 检查模型文件
    if not check_model_file():
        return

    # 初始化各模块
    try:
        camera = Camera(camera_id=0)
        preprocessor = VideoPreprocessor(flip=True, target_width=1280)
        hand_tracker = HandTracker(max_hands=2)
        display = Display()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ 所有模块初始化完成")
    print("💡 将双手放在摄像头前，系统会自动识别手部")
    print("💡 按 Q 键退出\n")

    # FPS计算
    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            # 1. 读取摄像头画面
            ret, frame = camera.read_frame()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break

            # 2. 预处理
            rgb_frame, bgr_frame = preprocessor.process(frame)

            # 3. 手部检测（新版API）
            results = hand_tracker.detect(rgb_frame)

            # 4. 提取关键点（用于后续分析）
            hands_data = hand_tracker.get_landmarks(results)

            # 5. 绘制骨架
            output_frame = display.draw_landmarks(bgr_frame, results)

            # 6. 计算FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # 7. 添加信息文字
            output_frame = display.add_info(output_frame, fps, len(hands_data))

            # 8. 显示画面
            display.show('iGuitar - Hand Tracking (Tasks API)', output_frame)

            # 9. 检查退出
            if display.wait_key(1) == ord('q'):
                print("\n👋 用户退出程序")
                break

            # 10. 调试信息（每30帧打印一次）
            frame_count += 1
            if hands_data and frame_count % 30 == 0:
                for hand_info in hands_data:
                    print(f"✋ 检测到 {hand_info['hand']} 手, "
                          f"关键点数: {len(hand_info['landmarks'])}, "
                          f"FPS: {int(fps)}")

    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")

    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        camera.release()
        hand_tracker.close()
        display.destroy_all()
        print("\n✅ 程序已安全退出")
        print(f"📊 总共处理了 {frame_count} 帧")


if __name__ == "__main__":
    main()