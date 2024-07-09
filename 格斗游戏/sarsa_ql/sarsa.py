import random
import numpy as np
from myenv.fightingice_env import FightingiceEnv
import scipy.io as io
from datetime import datetime


if __name__ == '__main__':
    env = FightingiceEnv(port=4242)

    gamma = 0.95
    alpha = 0.01
    epsilon = 0.1
    weight = np.zeros((40, 144))  # 0初始化

    n = 0
    p = 0
    RewardData = []
    N = 500  # 训练轮次
    NumWin = 0
    NumLose = 0
    Best_Score = 0
    abs_score = 0

    act = random.randint(0, 39)  # 随机初始动作

    while True:
        obs = env.reset()
        reward, done, info = 0, False, None
        n = n + 1
        r = 0
        a = datetime.now()  # 计时开始
        while not done:
            if n < N:
                if np.mod(n, 5) == 0:
                    string = "./SARSA_train/weight"+str(n+p)
                    io.savemat(string, {'weight': weight})
                    string = "./SARSA_train/reward"+str(n+p)
                    io.savemat(string, {'r': RewardData})  # 存储weight和当前reward信息
                if Best_Score <= abs_score:
                    Best_Score = abs_score
                    string = "./SARSA_train/weight_best"
                    io.savemat(string, {'weight': weight})
                    string = "./SARSA_train/reward_best"
                    io.savemat(string, {'r': RewardData})  # 存储最优weight和当前reward信息
            else:
                print("训练结束")
                string = "./SARSA_train/weight"+str(n+p)
                io.savemat(string, {'weight': weight})
                string = "./SARSA_train/reward"+str(n+p)
                io.savemat(string, {'r': RewardData})  # 存储最终weight和当前reward信息
                break

            act = np.argmax(np.dot(weight, obs))  # 观测量
            if random.random() < epsilon:
                act = random.randint(0, 39)
            else:
                pass  # epsilon-贪心策略

            # 动作采样
            new_obs, reward, done, info = env.step(act)

            if not done:
                # Sarsa学习过程
                explore_act = np.argmax(np.dot(weight, new_obs))

                if random.random() < epsilon:
                    explore_act = random.randint(0, 39)
                else:
                    pass  # epsilon-贪心策略

                delta = reward + gamma * np.dot(new_obs, weight[explore_act]) - np.dot(obs, weight[act])
                weight[act] = weight[act] + alpha * delta * obs  # weight更新过程
                obs = new_obs
                r = r + reward  # reward记录
            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0] > info[1] else 'lose'),
                      '训练局数', n)
                if info[0] > info[1]:
                    NumWin = NumWin + 1
                    abs_score = info[0] - info[1]
                elif info[0] < info[1]:
                    NumLose = NumLose + 1
            else:
                pass

        b = datetime.now()  # 计时结束
        print("game = {} time = {}".format(n, (b - a).seconds))
        if n % 50 == 0:
            win_rate = NumWin / (NumWin + NumLose)
            print("win rate = {}".format(win_rate))
            NumWin = 0
            NumLose = 0  # 每50轮次测试胜率
        if n == N:
            break

        RewardData.append(r)

    print("finish training")
