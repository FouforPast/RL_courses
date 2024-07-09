import sys
import time

import numpy as np
import env
import os
from base import Base


class SarsaLambda(Base):
    def __init__(self, num_epoch=20000, gamma=0.98, steps_total=300, num_angle=200, num_angle_v=200, lr=0.9, eps=0.9,
                 _lambda=0.9):
        super().__init__(f'results/sarsa_lambda/train_{_lambda}.log', num_epoch, gamma, steps_total, num_angle,
                         num_angle_v, lr, eps)
        self.qw_path = f'results/sarsa_lambda/QValues_{_lambda}.npy'
        self.track_path = f'results/sarsa_lambda/QTrack_{_lambda}.txt'
        self.log_path = 'results/sarsa_lambda/train.log'
        self.name = 'sarsa($\lambda$)算法'
        self.Q = np.zeros((3, self.num_angle, self.num_angle_v))
        self.steps_total = steps_total
        self._lambda = _lambda
        self.decay = 0.75
        if not os.path.exists(dir_path := os.path.dirname(self.qw_path)):
            os.mkdir(dir_path)

    def getEpsGreedyAction(self, eps, idx_angle, idx_angle_v):
        greedy_action = np.argmax(self.Q[:, idx_angle, idx_angle_v])
        if np.random.random() > eps:
            return greedy_action
        else:
            return np.random.choice(self.Q.shape[0])

    def decayLrEps(self, it, eps, lr):
        step_epoch1 = self.num_epoch // 10
        step_epoch2 = self.num_epoch // 20
        if (it + 1) % step_epoch1 == 0:
            lr = max(lr * self.decay, self.lr_min)
        if (it + 1) % step_epoch2 == 0:
            eps = max(eps * self.decay, self.eps_min)
        return eps, lr

    def trainCore(self, eps, lr, convergence_angle, convergence_angle_v):
        # 初始化
        angle, angle_v = self.initial_angle, self.initial_angle_v
        step, reward = 0, 0
        eligibility = np.zeros((3, self.num_angle, self.num_angle_v))
        # 到达结束状态时的误差
        error = np.abs((angle + np.pi) % (2 * np.pi) - np.pi)
        track = []  # 存储探索时的轨迹
        # 收敛限制
        error_convergence, reward_convergence = np.pi, -np.inf
        steps2stable = 0

        # 探索
        idx_angle, idx_angle_v = env.getDiscreteState(angle, angle_v, self.num_angle, self.num_angle_v)
        action = self.getEpsGreedyAction(eps, idx_angle, idx_angle_v)
        track.append((angle, angle_v, self.actions[action]))
        while step < self.steps_total:
            step += 1
            steps2stable += 1
            # 执行动作并获得观察量
            angle_new, angle_v_new = env.getNextState(angle, angle_v, self.actions[action])
            idx_angle_new, idx_angle_v_new = env.getDiscreteState(angle_new, angle_v_new, self.num_angle,
                                                                  self.num_angle_v)
            reward_step = env.getReward(angle, angle_v, self.actions[action])
            reward += reward_step
            # 选择新动作
            action_new = self.getEpsGreedyAction(eps, idx_angle_new, idx_angle_v_new)
            # 更新资格迹和delta
            eligibility[action, idx_angle, idx_angle_v] += 1
            delta = reward_step + self.gamma * self.Q[action_new, idx_angle_new, idx_angle_v_new] \
                    - self.Q[action, idx_angle, idx_angle_v]
            # 更新Q函数和资格迹
            self.Q += lr * delta * eligibility
            eligibility *= self.gamma * self._lambda
            # 更新状态和动作
            angle, angle_v, idx_angle, idx_angle_v = angle_new, angle_v_new, idx_angle_new, idx_angle_v_new
            action = action_new
            # 记录观察量
            track.append((angle, angle_v, self.actions[action]))
            # 如果新状态满足收敛条件，跳出循环
            error = min(error, np.abs(angle_new))
            if np.abs(angle_new) < convergence_angle and np.abs(angle_v_new) < convergence_angle_v:
                if steps2stable < 10:
                    error_convergence = error
                    reward_convergence = reward
                    break
                steps2stable = 0
        track.append((step, reward))
        return track, reward, step, error, reward_convergence, error_convergence


if __name__ == '__main__':
    f = SarsaLambda(num_epoch=30000, steps_total=300, lr=0.9, eps=0.9)
    # f.visualizeAction(f.qw_path)
    f.train()
