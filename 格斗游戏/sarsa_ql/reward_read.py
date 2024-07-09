import scipy.io as io


# 读取reward文件并存储于txt文档中
reward = io.loadmat('./SARSA_train/reward500')

with open("sarsa_reward.txt", "w") as f:
    f.write(str(reward))
