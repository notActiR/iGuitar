"""
和弦数据库：存储常用和弦的标准指法，并提供比对函数
"""

# 标准指法格式：{弦号: 品号}，弦号1~6（1最细，6最粗），品号0表示空弦
CHORDS = {
    'C':  {1: 0, 2: 1, 3: 0, 4: 2, 5: 3, 6: 0},   # 标准C和弦
    'G':  {1: 3, 2: 0, 3: 0, 4: 0, 5: 2, 6: 3},   # G和弦（简化版）
    'D':  {1: 2, 2: 3, 3: 2, 4: 0, 5: 0, 6: 0},   # D和弦
    'Am': {1: 0, 2: 1, 3: 2, 4: 2, 5: 0, 6: 0},   # Am和弦
    'Em': {1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 0},   # Em和弦
}

def check_chord(actual_fingers, chord_name):
    """
    比对实际手指位置与预期和弦
    :param actual_fingers: 字典，键为手指名，值为 (fret, string) 或 None
    :param chord_name: 和弦名（如 'C'）
    :return: 字典，键为弦号，值为状态字符串：
             'correct' - 正确按下
             'wrong'   - 按了但品不对
             'extra'   - 多按的弦
             'missing' - 应该按但没按（暂时未实现，需要根据手指判断哪根弦缺失）
             注意：本简化版本只标记了每根弦是否被正确按下，缺失检测需要更复杂的逻辑，
             这里暂不实现，留待后续扩展。
    """
    expected = CHORDS[chord_name]
    result = {}
    # 初始化所有弦为 'missing'（实际程序可能标记为未按）
    # 但更合理的做法是：如果某根弦预期有品，而实际没有手指按它，就是缺失。
    # 由于 actual_fingers 只包含有手指的弦，我们无法直接知道哪些弦没有手指。
    # 所以这里我们先标记实际有手指的弦，缺失逻辑稍后可以在UI中提示。
    for string in range(1, 7):
        result[string] = 'unknown'  # 未知状态

    # 处理实际按下的手指
    for finger, pos in actual_fingers.items():
        if pos is None:
            continue
        fret, string = pos
        expected_fret = expected.get(string, 0)
        if expected_fret == fret:
            result[string] = 'correct'
        else:
            # 如果该弦预期为非空弦，但按错品 → wrong
            # 如果该弦预期为空弦，却按了 → extra
            if expected_fret != 0:
                result[string] = 'wrong'
            else:
                result[string] = 'extra'

    return result