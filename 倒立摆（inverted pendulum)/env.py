import numpy as np

m = 0.055
g = 9.81
l = 0.042
J = 1.91e-4
b = 3e-6
K = 0.0536
R = 9.5
Ts = 0.005
v_min = -15 * np.pi
v_max = 15 * np.pi


def getNextState(angle, angle_v, u):
    """
    获取下一时刻的状态
    :param angle: 当前角度
    :param angle_v: 当前角速度
    :param u: 输入电压
    :return: 下一时刻角度，下一时刻角速度
    """
    angle_new = angle + Ts * angle_v
    angle_a = (m * g * l * np.sin(angle) - b * angle_v - K ** 2 / R * angle_v + K / R * u) / J
    angle_v_new = angle_v + Ts * angle_a
    # 正则化角度
    angle_new = (angle_new + np.pi) % (2 * np.pi) - np.pi
    # 加上速度限制
    angle_v_new = max(angle_v_new, v_min) if angle_v_new < 0 else min(angle_v_new, v_max)
    if angle_new > np.pi or angle_new < -np.pi:
        raise Exception("fsdhkjf")
    if angle_v_new > v_max or angle_v_new < v_min:
        raise Exception("fsdhkjf")
    return angle_new, angle_v_new


def getReward(angle, angle_v, u):
    """
    奖励函数
    :param angle: 当前角度
    :param angle_v: 当前角速度
    :param u: 输入电压
    :return: 奖励
    """
    return -angle ** 2 * 5 - angle_v ** 2 * 0.1 - u ** 2


def getDiscreteState(angle, angle_v, num_angle=200, num_angle_v=200):
    """
    获取当前状态对应的离散化状态编号
    :param angle: 当前角度
    :param angle_v: 当前角速度
    :param num_angle: 角度的离散化数量
    :param num_angle_v: 角速度的离散化数量
    :return: 离散化的坐标
    """
    idx_angle = int((angle + np.pi) / (2 * np.pi) * num_angle)
    idx_angle_v = int((angle_v - v_min) / (v_max - v_min) * num_angle_v)
    idx_angle = min(idx_angle, num_angle - 1)
    idx_angle_v = min(idx_angle_v, num_angle_v - 1)
    return idx_angle, idx_angle_v
