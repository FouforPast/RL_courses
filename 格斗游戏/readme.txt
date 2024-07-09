为了减小打包后的文件大小，已经将文件夹中的data文件夹删除，如果需要，可将老师提供的mindspore_dqn中的data文件夹复制到相应文件夹中。
replay文件夹中存放一个回放文件以及对应的录制视频，其余文件夹存放不同算法的代码文件
--------------------BY 于海琛---------------------
代码目录：sarsa_ql
Sarsa：
运行sarsa.py可实现Sarsa算法的训练过程，相关权重和Reward信息存储在SARSA_train文件夹下。

Docker指令为：
docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_sarsa fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python sarsa.py > ./logs/sarsa_log1.txt 2> ./logs/sarsa_error_log1.txt'

指令含义为：继承fighting_game:v1，建立一个运行结束即删除的Docker容器“fighting_game_sarsa”，在其中运行sarsa.py，并将运行log存储在logs文件夹下的sarsa_log1.txt文档中，错误信息存储在logs文件夹下的sarsa_error_log1.txt文档中。

将test.py中第20行的权重路径设置为“./SARSA_train/weight_best”，并运行test.py即可实现Sarsa算法的测试过程。

Docker指令为：
docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_sarsa fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python test.py > ./logs/sarsa_test1.txt 2> ./logs/sarsa_error_test1.txt'

指令含义为：继承fighting_game:v1，建立一个运行结束即删除的Docker容器“fighting_game_sarsa”，在其中运行test.py，并将运行log存储在logs文件夹下的sarsa_test1.txt文档中，错误信息存储在logs文件夹下的sarsa_error_test1.txt文档中。



Q-Learning：
运行QL.py可实现Q-Learning算法的训练过程，相关权重和Reward信息存储在QL_train文件夹下。

Docker指令为：
docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_ql fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python QL.py > ./logs/ql_log1.txt 2> ./logs/ql_error_log1.txt'

指令含义为：继承fighting_game:v1，建立一个运行结束即删除的Docker容器“fighting_game_ql”，在其中运行QL.py，并将运行log存储在logs文件夹下的ql_log1.txt文档中，错误信息存储在logs文件夹下的ql_error_log1.txt文档中。

将test.py中第20行的权重路径设置为“./QL_train/weight_best”，并运行test.py即可实现Q-Learning算法的测试过程。

Docker指令为：
docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_ql fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python test.py > ./logs/ql_test1.txt 2> ./logs/ql_error_test1.txt'

指令含义为：继承fighting_game:v1，建立一个运行结束即删除的Docker容器“fighting_game_ql”，在其中运行test.py，并将运行log存储在logs文件夹下的ql_test1.txt文档中，错误信息存储在logs文件夹下的ql_error_test1.txt文档中。



其他：
想要将某个.mat权重文件读取到.txt文档中以方便打开，可以更改reward_read.py中第五行的路径信息并运行，即可得到方便直接读取的txt文档。

--------------------BY 李英龙---------------------
代码目录：dqn，dqn_failed
dqn文件夹是double dueling dqn的代码；dqn_failed文件夹是一个失败算法的代码。
训练的话直接运行start_train.sh即可；
测试的话运行start_eval.sh会失败，需要
将src/config.py中的
'lr': 0.001改为'lr': 0.00000000001
'num_evaluate_episode': 3改为'num_evaluate_episode': 1000,
将train.py中的
eval_cb = MyEvaluateCallback(10)改为eval_cb = MyEvaluateCallback(1)，
episode=1000改为episode=2