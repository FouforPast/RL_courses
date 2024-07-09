#!/bin/bash
cd $(dirname $0)
docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_2 fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python train.py > train_log_2.txt 2> train_error_log_2.txt'
# docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_3 fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python train.py > train_log_3.txt 2> train_error_log_3.txt'

# docker run -itd -v "${PWD}":/app -w /app --name fighting_game2 fighting_game:v1 /bin/sh
# 这个命令是测试Java环境是否运行正常使用
# docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; bash ftg.sh'