import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches

import env
from base import Base

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

class QLearningSGD(Base):
    def __init__(self, num_epoch=20000, gamma=0.98, steps_total=300, num_angle=200, num_angle_v=200, lr=0.9, eps=0.9):
        super().__init__('results/QLearning_sgd/train.log', num_epoch, gamma, steps_total, num_angle, num_angle_v, lr,
                         eps)
        self.qw_path = 'results/QLearning_sgd/wValues.npy'
        self.track_path = 'results/QLearning_sgd/QTrack.txt'
        self.log_path = 'results/QLearning_sgd/train.log'
        self.num = 9
        self.name = '随机梯度下降的QLearning算法'
        self.w = np.zeros((self.num, self.num, 3))
        self.steps_total = steps_total
        if not os.path.exists(dir_path := os.path.dirname(self.qw_path)):
            os.mkdir(dir_path)
        self.sigma_angle = 2 * np.pi / (self.num - 1)
        self.sigma_angle_v = 30 * np.pi / (self.num - 1)
        self.cov = np.array([[self.sigma_angle ** 2, 0], [0, self.sigma_angle_v ** 2]])
        self.mu_angle = np.arange(-np.pi + np.pi / self.num, np.pi, 2 * np.pi / self.num)
        self.mu_angle_v = np.arange(env.v_min + (env.v_max - env.v_min) / (2 * self.num),
                                    env.v_max, (env.v_max - env.v_min) / self.num)
        self.state_mu = np.transpose(np.array(np.meshgrid(self.mu_angle, self.mu_angle_v)), (1, 2, 0))

    def getState(self, angle, angle_v):
        s = np.stack([angle, angle_v], axis=-1)[..., np.newaxis, np.newaxis, :]
        x1 = (s - self.state_mu)[..., np.newaxis, :]
        x2 = (s - self.state_mu)[..., :, np.newaxis]
        x1 = x1 @ np.linalg.inv(self.cov)
        rtn = np.exp(-0.5 * np.squeeze(x1 @ x2))
        rtn = rtn / np.sum(rtn)
        return rtn

    def getQ(self, angle, angle_v, action=None):
        state = self.getState(angle, angle_v)
        if action is None:
            q0 = np.sum(state * self.w[:, :, 0], axis=(-2, -1))
            q1 = np.sum(state * self.w[:, :, 1], axis=(-2, -1))
            q2 = np.sum(state * self.w[:, :, 2], axis=(-2, -1))
            return np.array([q0, q1, q2])
        else:
            return np.sum(state * self.w[:, :, action], axis=(-2, -1))

    def getEpsGreedyAction(self, eps, angle, angle_v):
        q = self.getQ(angle, angle_v)
        greedy_action = np.argmax(q)
        if np.random.random() > eps:
            return greedy_action
        else:
            return np.random.choice(q.shape[0])

    def trainCore(self, eps, lr, convergence_angle, convergence_angle_v):
        # 初始化
        angle, angle_v = self.initial_angle, self.initial_angle_v
        step, reward = 0, 0
        # 到达结束状态时的误差
        error = np.abs((angle + np.pi) % (2 * np.pi) - np.pi)
        track = []  # 存储探索时的轨迹
        error_convergence, reward_convergence = np.pi, -np.inf
        steps2stable = 0

        # 探索
        while step < self.steps_total:
            step += 1
            steps2stable += 1
            # 采样动作并执行
            action = self.getEpsGreedyAction(eps, angle, angle_v)
            # 获得观测量和奖励
            angle_new, angle_v_new = env.getNextState(angle, angle_v, self.actions[action])
            reward_step = env.getReward(angle, angle_v, self.actions[action])
            # 记录状态、动作和奖励
            track.append((angle, angle_v, self.actions[action], reward_step, angle_new, angle_v_new))
            reward += reward_step
            error = min(error, np.abs(angle_new))
            # 计算TD误差
            q_max = np.max(self.getQ(angle_new, angle_v_new))
            delta = reward_step + self.gamma * q_max - self.getQ(angle, angle_v, action)
            # 更新权重
            self.updateW(lr, delta, angle, angle_v, action)
            # 更新状态
            angle, angle_v = angle_new, angle_v_new
            # 如果新状态满足收敛条件，跳出循环
            if np.abs(angle_new) < convergence_angle and np.abs(angle_v_new) < convergence_angle_v:
                if steps2stable < 10:
                    error_convergence = error
                    reward_convergence = reward
                    break
                steps2stable = 0
        track.append((step, reward))
        return track, reward, step, error, reward_convergence, error_convergence

    def decayLrEps(self, it, eps, lr):
        step_epoch1 = self.num_epoch // 6
        step_epoch2 = self.num_epoch // 8
        if (it + 1) % step_epoch1 == 0:
            lr = max(lr * self.decay, self.lr_min)
        if (it + 1) % step_epoch2 == 0:
            eps = max(eps * self.decay, self.eps_min)
        return eps, lr

    def visualizeAction(self, file_path='results/QLearning_sgd/wValues.npy', num_bins1=300, num_bins2=300):
        """
        可视化动作
        :return:
        """
        self.w = np.load(file_path)
        q_action = self.getQAction(num_bins1, num_bins2)
        fig = plt.figure(figsize=(16, 8))
        colormap = colors.ListedColormap(["black", "darkblue", "lightblue", "yellow"])
        plt.imshow(q_action, cmap=colormap)
        plt.yticks(np.arange(0, num_bins1 + 0.1, num_bins1 // 8),
                   list(map(lambda x: str(round(x, 4))[:6], list(np.arange(-np.pi, np.pi + 0.01, np.pi / 4)))),
                   fontsize=6)
        plt.xticks(np.arange(0, num_bins2 + 0.1, num_bins2 // 8),
                   list(map(lambda x: str(round(x, 4))[:6],
                            list(np.arange(env.v_min, env.v_max + 0.1, (env.v_max - env.v_min) / 8)))),
                   fontsize=6)
        patch = [patches.Patch(color='black', label="random"), patches.Patch(color='darkblue', label="-3V"),
                 patches.Patch(color='lightblue', label="0V"), patches.Patch(color='yellow', label="3V")]
        plt.xlabel('diff_alpha rad/s')
        plt.ylabel('alpha rad')
        plt.title('随机梯度下降的QLearning算法的动作选择')
        plt.legend(handles=patch, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        fig.savefig(os.path.dirname(file_path) + '/action.jpg', dpi=300)

    def getQAction(self, num_bins1=200, num_bins2=200):
        if self.w is None:
            return

        # 定义三维数据
        xx = np.arange(-np.pi, np.pi, 2 * np.pi / num_bins1)
        yy = np.arange(env.v_min, env.v_max, (-env.v_min + env.v_max) / num_bins2)

        # 生成网格点坐标矩阵,对x和y数据执行网格化
        X, Y = np.meshgrid(xx, yy)
        Z = self.getQ(X, Y)
        q_action = np.zeros([num_bins1, num_bins2])
        q_action -= 1
        q_action[(Z[0] > Z[1]) & (Z[0] > Z[2])] = 0
        q_action[(Z[1] > Z[0]) & (Z[1] > Z[2])] = 1
        q_action[(Z[2] > Z[1]) & (Z[2] > Z[0])] = 2
        # angle_bin = 2 * np.pi / num_bins1
        # angle_bin_v = (env.v_max - env.v_min) / num_bins2
        # for i in range(q_action.shape[0]):
        #     for j in range(q_action.shape[1]):
        #         q = self.getQ(-np.pi + i * angle_bin, env.v_min + j * angle_bin_v)
        #         if q[0] == q[1] == q[2]:
        #             continue
        #         q_action[i, j] = np.argmax(q)
        q_action[0, 0] = -1
        return q_action

    def updateW(self, lr, delta, angle, angle_v, action):
        state = self.getState(angle, angle_v)
        self.w[:, :, action] += lr * delta * state

    def visualizeQ(self, file_path):
        """
        可视化Q函数
        :param file_path: 存储Q值的路径
        :return:
        """
        import matplotlib.pyplot as plt
        if not os.path.exists(file_path):
            return
        self.w = np.load(file_path)
        dirname = os.path.dirname(file_path)

        # 定义三维数据
        xx = np.arange(-np.pi, np.pi, 2 * np.pi / 500)
        yy = np.arange(env.v_min, env.v_max, (-env.v_min + env.v_max) / 500)

        # 生成网格点坐标矩阵,对x和y数据执行网格化
        X, Y = np.meshgrid(xx, yy)

        # 定义新坐标轴
        for i in range(3):
            fig = plt.figure(figsize=(10, 10))
            ax3 = plt.axes(projection='3d')
            # 计算z轴数据
            Z = self.getQ(X, Y, action=i)
            # 绘图
            # 函数plot_surface期望其输入结构为一个规则的二维网格
            ax3.plot_surface(X, Y, Z, cmap='rainbow')  # cmap是颜色映射表
            plt.title(f'Q({i},:,:)', fontdict={'size': 20})
            ax3.set_xlabel('$angle \quad rad$', fontdict={'size': 8})
            ax3.set_ylabel('$angle_v \quad rad/s$', fontdict={'size': 8})
            ax3.set_zlabel('$Q value$', fontdict={'size': 8})
            plt.subplots_adjust()
            plt.tight_layout()
            fig.savefig(f'{dirname}/Q_action={self.actions[i]}.jpg', dpi=300, bbox_inches='tight')
            plt.show()
        # fig = plt.figure()
        # ax3 = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132, projection='3d'), fig.add_subplot(133, projection='3d')]
        # # 定义新坐标轴
        # for i in range(3):
        #     # fig = plt.figure(figsize=(10, 10))
        #     # ax3 = plt.axes(projection='3d')
        #     # 计算z轴数据
        #     Z = self.getQ(X, Y, action=i)
        #     # 绘图
        #     # 函数plot_surface期望其输入结构为一个规则的二维网格
        #     ax3[i].plot_surface(X, Y, Z, cmap='rainbow')  # cmap是颜色映射表
        #     plt.title(f'Q({i},:,:)', fontdict={'size': 20})
        #     ax3[i].set_xlabel('$angle \quad rad$', fontdict={'size': 8})
        #     ax3[i].set_ylabel('$angle_v \quad rad/s$', fontdict={'size': 8})
        #     ax3[i].set_zlabel('$Q value$', fontdict={'size': 8})
        #     plt.subplots_adjust()
        #     plt.tight_layout()
        # fig.savefig(f'{dirname}/Q_action={self.actions[i]}.jpg', dpi=300, bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    f = QLearningSGD(num_epoch=30000, steps_total=300, lr=0.9, eps=0.9)
    # f.visualizeQ(ql.qw_path)
    # f.visualizeAction()
    f.train()
