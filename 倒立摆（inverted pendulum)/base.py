import logging
import os.path
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anima
from matplotlib import patches, colors

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


class Base:
    def __init__(self, log_path, num_epoch=20000, gamma=0.98, steps_total=300, num_angle=200, num_angle_v=200, lr=0.9,
                 eps=0.9):
        self.log_path = None
        self.lr = lr
        self.eps = eps
        self.track_path = None
        self.qw_path = None
        self.initial_angle, self.initial_angle_v = -np.pi, 0  # 每轮探索时的初始状态
        self.decay = 0.566  # 衰减率
        self.lr_min = 5e-2
        self.eps_min = 1e-3
        self.log = None
        self.num = 20
        self.num_epoch = num_epoch
        self.gamma = gamma
        self.steps_total = steps_total
        self.num_angle = num_angle
        self.num_angle_v = num_angle_v
        self.actions = [-3, 0, 3]
        self.Q = None
        self.w = None
        self.name = None
        # 创建logger对象
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)

        # 创建文件handler并设置级别为DEBUG
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台handler并设置级别为INFO
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 将格式化器添加到handler中
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将handler添加到logger对象中
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def showTrack(self, track=None, file_path=None):
        """
        画出轨迹的图形
        :param track: 轨迹
        :param file_path: 存储轨迹的路径
        :return: None
        """
        dirname = ''
        if file_path is not None:
            try:
                with open(file_path, 'r', encoding='utf8') as f:
                    track = [line.strip().split() for line in f.readlines()]
                dirname = os.path.dirname(file_path)
            except FileNotFoundError as e:
                return
        if track is None:
            return
        _, _ = track[-1]
        track = np.array(track[:-1], dtype=np.float)
        angles = track[:, 0]
        fig = plt.figure()
        plt.title("Inverted Pendulum")
        plt.axis('off')
        images = []
        for a in angles:
            img = plt.plot((0, np.sin(a)), (0, np.cos(a)), color='r')
            images.append(img)
        ani = anima.ArtistAnimation(fig, images, interval=5, repeat_delay=1000)
        ani.save(f"{dirname}/state_visualization.gif", writer='pillow')

    def save(self, track, file_path1, file_path2):
        """
        保存轨迹文件和Q值
        :param track: 轨迹
        :param file_path1: Q值保存路径
        :param file_path2: 轨迹保存路径
        :return: None
        """
        if self.Q is not None:
            np.save(file_path1, self.Q)
        else:
            np.save(file_path1, self.w)
        with open(file_path2, 'w', encoding='utf-8') as f:
            for t in track:
                for item in t:
                    f.write(str(item) + ' ')
                f.write('\n')

    def decayLrEps(self, it, eps, lr):
        return eps, lr

    def train(self):
        t1 = time.time()
        num_epoch = self.num_epoch
        self.actions = [-3, 0, 3]  # 动作离散化
        eps = self.eps  # eps贪心策略
        lr = self.lr  # 学习率
        convergence_angle, convergence_angle_v = 0.05, 0.01  # 收敛角度和角速度控制
        error_min, reward_max, steps_min = np.pi, -np.inf, self.steps_total
        error_final, reward_final = np.pi, -np.inf
        min_epoch = -1  # 最早收敛epoch
        step_convergence = -1  # 收敛步数

        rewards = []

        for it in range(num_epoch):
            # 更新eps和学习率
            eps, lr = self.decayLrEps(it, eps, lr)
            track, reward, step, error, reward_convergence, error_convergence = \
                self.trainCore(eps, lr, convergence_angle, convergence_angle_v)

            steps_min = min(steps_min, step)
            error_min = min(error, error_min)
            reward_max = max(reward_max, reward)
            # 如果本次探索收敛且收敛时的累积奖励更高，存储本次轨迹
            if step < self.steps_total:
                # if error_convergence < error_final or error_convergence == error_final and reward_convergence > reward_final:
                if reward_convergence > reward_final:
                    error_final, reward_final = error_convergence, reward_convergence
                    min_epoch = it if min_epoch == -1 else min_epoch
                    self.save(track, self.qw_path, self.track_path)
                    step_convergence = step
            # 打印本次探索信息
            if (it + 1) % 100 == 0:
                self.logger.info(
                    f'第{it + 1}次迭代，探索步长限制{self.steps_total}，角度最小误差{error_min}，最大累积奖励{reward_max}, '
                    f'最小步数{steps_min}, eps={eps}, lr={lr}')
                rewards.append(reward_max)
                error_min, reward_max, steps_min = np.pi, -np.inf, self.steps_total
        self.logger.info(f'训练总轮次{self.num_epoch}，最小角度误差{error_final}, 最大奖励{reward_final}, '
                         f'最早收敛轮次{min_epoch}，到达收敛状态最小步数{step_convergence}，训练耗时{time.time() - t1}')
        self.showTrack(file_path=self.track_path)
        self.showReward(rewards, os.path.dirname(self.track_path))
        self.visualizeQ(self.qw_path)
        self.visualizeAction(self.qw_path)
        # sys.stdout.close()

    def showReward(self, rewards, dir_path):
        fig = plt.figure()
        plt.plot(np.arange(0, len(rewards) * 100, 100), rewards)
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.title(self.name + '的实验结果')
        fig.savefig(os.path.join(dir_path, 'rewards.jpg'), dpi=300, bbox_inches='tight')
        plt.show()

    def visualizeAction(self, file_path):
        """
        可视化动作
        :return:
        """
        self.Q = np.load(file_path)
        q_action = self.getQAction()
        fig = plt.figure(figsize=(16, 8))
        colormap = colors.ListedColormap(["black", "darkblue", "lightblue", "yellow"])
        plt.imshow(q_action, cmap=colormap)
        plt.yticks(np.arange(0, 200.01, 25),
                   list(map(lambda x: str(round(x, 4))[:6], list(np.arange(-np.pi, np.pi + 0.01, np.pi / 4)))),
                   fontsize=6)
        plt.xticks([0, 33, 66, 99, 133, 167, 200], ['-15pi', '-10*pi', '-5*pi', 0, '5*pi', '10*pi', '15pi'])
        patch = [patches.Patch(color='black', label="random"), patches.Patch(color='darkblue', label="-3V"),
                 patches.Patch(color='lightblue', label="0V"), patches.Patch(color='yellow', label="3V")]
        plt.xlabel('angle_v rad/s')
        plt.ylabel('angle rad')
        plt.title(self.name + '的动作选择')
        plt.legend(handles=patch, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        fig.savefig(os.path.dirname(file_path) + '/action.jpg', dpi=300)

    def getQAction(self, num_bins1=200, num_bins2=200):
        if self.Q is None:
            return
        q_action = np.zeros([self.num_angle, self.num_angle_v])
        q_action -= 1
        for i in range(q_action.shape[0]):
            for j in range(q_action.shape[1]):
                if self.Q[0, i, j] == self.Q[1, i, j] == self.Q[2, i, j]:
                    continue
                q_action[i, j] = np.argmax(self.Q[:, i, j])
        return q_action

    def visualizeQ(self, file_path):
        """
        可视化Q函数
        :param file_path: 存储Q值的路径
        :return:
        """
        if not os.path.exists(file_path):
            return
        Q = np.load(file_path)
        dirname = os.path.dirname(file_path)

        # 定义三维数据
        xx = np.arange(0, self.num_angle, 1)
        yy = np.arange(0, self.num_angle_v, 1)

        # 生成网格点坐标矩阵,对x和y数据执行网格化
        X, Y = np.meshgrid(xx, yy)

        # 定义新坐标轴
        for i in range(3):
            fig = plt.figure(figsize=(10, 10))
            ax3 = plt.axes(projection='3d')
            # 计算z轴数据
            Z = Q[i, X, Y]
            # 绘图
            # 函数plot_surface期望其输入结构为一个规则的二维网格
            ax3.plot_surface(X, Y, Z, cmap='rainbow')  # cmap是颜色映射表
            plt.title(f'{self.name}:Q({i},:,:)')
            ax3.set_xlabel('$angle \quad rad$', fontdict={'size': 8})
            ax3.set_ylabel('$angle_v \quad rad/s$', fontdict={'size': 8})
            ax3.set_zlabel('$Q value$', fontdict={'size': 8})
            ax3.set_xticks(np.arange(0, 200.01, 25),
                           list(map(lambda x: str(round(x, 4))[:6], list(np.arange(-np.pi, np.pi + 0.01, np.pi / 4)))),
                           fontsize=6)
            ax3.set_yticks([0, 33, 66, 99, 133, 167, 200], ['-15pi', '-10*pi', '-5*pi', 0, '5*pi', '10*pi', '15pi'],
                           fontsize=6)
            # ax3.tick_params(axis='z', labelsize=6)
            plt.subplots_adjust()
            # plt.tight_layout()
            fig.savefig(f'{dirname}/Q_action={self.actions[i]}.jpg', dpi=200, bbox_inches='tight')
            # plt.show()

    def trainCore(self, eps, lr, convergence_angle, convergence_angle_v):
        pass


if __name__ == '__main__':
    # Base().visualizeQ(r"results/sarsa\QValues.npy")
    Base().visualizeAction()
