from sarsa import Sarsa
from sarsa_lambda import SarsaLambda
from Qlearning import QLearning
from QLearning_sgd import QLearningSGD

if __name__ == '__main__':

    # Sarsa
    f = Sarsa(num_epoch=30000, steps_total=300, lr=0.9, eps=0.9)

    # SarsaLambda
    f = SarsaLambda(num_epoch=30000, steps_total=300, lr=0.9, eps=0.9)

    # Q-Learning
    f = QLearning(num_epoch=30000, steps_total=300, lr=0.9, eps=0.9)

    # 随机梯度下降的Q-Learning
    f = QLearningSGD(num_epoch=30000, steps_total=300, lr=0.9, eps=0.9)
    # f.visualizeQ(f.qw_path)
    # f.visualizeAction(f.qw_path)
    
    f.train()