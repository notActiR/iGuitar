"""公共绘图函数模块"""
import cv2

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
