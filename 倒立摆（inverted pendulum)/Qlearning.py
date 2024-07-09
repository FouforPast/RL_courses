import sys

import numpy as np
import env
import os
import time
from base import Base


class QLearning(Base):
    def __init__(self, num_epoch=20000, gamma=0.98, steps_total=300, num_angle=200, num_angle_v=200, lr=0.9, eps=0.9):
        super().__init__('results/QLearning/train.log', num_epoch, gamma, steps_total, num_angle, num_angle_v, lr, eps)
        self.qw_path = 'results/QLearning/QValues.npy'
        self.track_path = 'results/QLearning/QTrack.txt'
        self.log_path = 'results/QLearning/train.log'
        self.name = 'QLearning算法'
        self.Q = np.zeros((3, self.num_angle, self.num_angle_v))
        self.steps_total = steps_total
        if not os.path.exists(dir_path := os.path.dirname(self.qw_path)):
            os.mkdir(dir_path)

    def getEpsGreedyAction(self, eps, idx_angle, idx_angle_v):
        greedy_action = np.argmax(self.Q[:, idx_angle, idx_angle_v])
        if np.random.random() > eps:
            return greedy_action
        else:
            return np.random.choice(self.Q.shape[0])

    def visualizeAction(self, file_path='results/QLearning/QValues.npy'):
        """
        可视化动作
        :return:
        """
        super().visualizeAction(file_path)

    def decayLrEps(self, it, eps, lr):
        step_epoch1 = self.num_epoch // 5
        step_epoch2 = self.num_epoch // 8
        if (it + 1) % step_epoch1 == 0:
            lr = max(lr * self.decay, self.lr_min)
        if (it + 1) % step_epoch2 == 0:
            eps = max(eps * self.decay, self.eps_min)
        return eps, lr

    def trainCore(self, eps, lr, convergence_angle, convergence_angle_v):
        # 初始化
        angle, angle_v = self.initial_angle, self.initial_angle_v
        step, reward = 0, 0
        # 到达结束状态时的误差
        error = np.abs((angle + np.pi) % (2 * np.pi) - np.pi)
        error_convergence, reward_convergence = np.pi, -np.inf
        track = []  # 存储探索时的轨迹
        steps2stable = 0
        # 探索
        idx_angle, idx_angle_v = env.getDiscreteState(angle, angle_v, self.num_angle, self.num_angle_v)
        while step < self.steps_total:
            step += 1
            steps2stable += 1
            # 采样动作并执行
            action = self.getEpsGreedyAction(eps, idx_angle, idx_angle_v)
            # 获得观测量和奖励
            angle_new, angle_v_new = env.getNextState(angle, angle_v, self.actions[action])
            reward_step = env.getReward(angle, angle_v, self.actions[action])
            # 记录状态、动作和奖励
            track.append((angle, angle_v, self.actions[action], reward_step, angle_new, angle_v_new))
            reward += reward_step
            error = min(error, np.abs(angle_new))
            # 更新Q
            idx_angle_new, idx_angle_v_new = env.getDiscreteState(angle_new, angle_v_new, self.num_angle, self.num_angle_v)
            delta = reward_step + self.gamma * np.max(self.Q[:, idx_angle_new, idx_angle_v_new]) \
                    - self.Q[action, idx_angle, idx_angle_v]
            self.Q[action, idx_angle, idx_angle_v] += lr * delta
            # 更新状态
            angle, angle_v, idx_angle, idx_angle_v = angle_new, angle_v_new, idx_angle_new, idx_angle_v_new
            # 如果新状态满足收敛条件，跳出循环
            if np.abs(angle_new) < convergence_angle and np.abs(angle_v_new) < convergence_angle_v:
                if steps2stable < 10:
                    error_convergence = error
                    reward_convergence = reward
                    break
                steps2stable = 0
        track.append((step, reward))
        return track, reward, step, error, reward_convergence, error_convergence


if __name__ == '__main__':
    f = QLearning(num_epoch=30000, steps_total=300, lr=0.9, eps=0.9)
    # f.visualizeAction(f.qw_path)
    f.train()