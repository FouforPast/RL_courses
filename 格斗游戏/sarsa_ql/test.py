import random
import numpy as np
from myenv.fightingice_env import FightingiceEnv
import scipy.io as io
import sys


def loadweight(weight_file=None):

    weight = io.loadmat(weight_file)
    weight = weight['weight']
    weight = weight.reshape(40, 144)

    return weight


if __name__ == '__main__':
    env = FightingiceEnv(port=4242)

    weight = loadweight("./QL_train/weight_best")  # 加载权重

    n = 0
    N = 30  # 测试轮次数
    Win = 0

    while True:
        obs = env.reset()
        reward, done, info = 0, False, None  # 初始设置
        n = n + 1  # 局数+1

        while not done:
            act = np.argmax(np.dot(weight, obs))
            new_obs, reward, done, info = env.step(act)

            if not done:
                obs = new_obs

            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0] > info[1] else 'lose'),
                      '测试局数', n)
                print('last reward', reward)
                if info[0] > info[1]:
                    Win = Win + 1
            else:
                exit('done')
                pass

        if n == N:
            print("WINNING TIMES", Win)
            break

    print("finish testing")
