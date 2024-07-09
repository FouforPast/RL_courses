import re
import numpy as np
from scipy.signal import savgol_filter

name = 'train_log_valid.txt'
text = open(name, 'r').read()

train_info = []


# 提取own_hp和opp_hp的数字
train_text = re.compile(r"At the end, own_hp (?P<own_hp>\d+): opp_hp (?P<opp_hp>\d+)\. .*\.\nEpisode (?P<eposide>\d+): loss is (?P<loss>-?\d+\.\d+), rewards is (?P<reward>-?\d+\.\d+)")
for i in train_text.finditer(text):
    train_info.append((i.group('own_hp'), i.group('opp_hp'), i.group('eposide'), i.group('loss'), i.group('reward')))

own_hps_val, opp_hps_val = [], []
val_text = re.compile(r"At the end, own_hp (?P<hp1>\d+): opp_hp (?P<hp2>\d+)\. .*\.\nAt the end, own_hp (?P<hp3>\d+): opp_hp (?P<hp4>\d+)\. you (win|lose)\.\nAt the end, own_hp (?P<hp5>\d+): opp_hp (?P<hp6>\d+)\. you (win|lose)")
for i in val_text.finditer(text):
    own_hps_val.extend([int(i.group('hp1')), int(i.group('hp3')), int(i.group('hp5'))])
    opp_hps_val.extend([int(i.group('hp2')), int(i.group('hp4')), int(i.group('hp6'))])


train_info2 = []
train_text2 = re.compile(r"Episode (?P<eposide>\d+): loss is (?P<loss>-?\d+\.\d+), rewards is (?P<reward>-?\d+\.\d+)")
for i in train_text2.finditer(text):
    train_info2.append((0, 0, i.group('eposide'), i.group('loss'), i.group('reward')))

val_info2 = []
val_text2 = re.compile(r"Evaluate for episode (?P<eposide>\d+) total rewards is (?P<reward>-?\d+\.\d+)")
for i in val_text2.finditer(text):
    val_info2.append((i.group('eposide'), i.group('reward')))


own_hps = np.array([int(_[0]) for _ in train_info])
opp_hps = np.array([int(_[1]) for _ in train_info])
eposides0 = np.array([int(_[2]) for _ in train_info])

eposides1 = np.array([int(_[2]) for _ in train_info2])
losses = np.array([float(_[3]) for _ in train_info2])
rewards = np.array([float(_[4]) for _ in train_info2])

own_hps_val = np.array(own_hps_val)
opp_hps_val = np.array(opp_hps_val)



eposides3 = np.array([int(_[0]) for _ in val_info2])
rewards2 = np.array([float(_[1]) for _ in val_info2])


import matplotlib.pyplot as plt
window_size1, window_size2, window_size3 = 200, 50, 20
polyorder = 4

plt.figure(figsize=(20, 10))
plt.plot(eposides0, own_hps - opp_hps, label='the difference of hp')
smooth = savgol_filter(own_hps - opp_hps, window_size1, polyorder)
plt.plot(eposides0, smooth, linestyle='--', lw=4, label='smooth of the difference of hp')
plt.legend()
plt.xlabel('eposides')
plt.ylabel('the difference of hp during training')
# plt.show()
plt.savefig('png_file/dqn_hp_train.png')

plt.figure(figsize=(20, 10))
smooth = savgol_filter(rewards, window_size1, polyorder)
plt.plot(eposides1, rewards, label='rewards')
plt.plot(eposides1, smooth, linestyle='--', lw=4, label='smooth of rewards')
plt.ylim(-20, 30)
plt.legend()
plt.xlabel
plt.ylabel('train rewards')
# plt.show()
plt.savefig('png_file/dqn_reward_train.png')

plt.figure(figsize=(20, 10))
smooth = savgol_filter(losses, window_size1, polyorder)
plt.plot(eposides1, losses, label='loss')
plt.plot(eposides1, smooth, linestyle='--', lw=4, label='smooth of loss')
plt.ylim(0, 2)
plt.legend()
plt.xlabel('eposides')
plt.ylabel('loss')
# plt.show()
plt.savefig('png_file/dqn_loss_train.png')


plt.figure(figsize=(20, 10))
data = own_hps_val - opp_hps_val
smooth = savgol_filter(data, window_size2, polyorder)
plt.plot(np.arange(data.shape[0]) * 10 / 3, data, label='the difference of hp')
plt.plot(np.arange(data.shape[0]) * 10 / 3, smooth, linestyle='--', lw=4, label='smooth of the difference of hp')
plt.legend()
plt.xlabel('eposides')
plt.ylabel('the difference of hp during evaluating')
# plt.show()
plt.savefig('png_file/dqn_hp_val.png')


plt.figure(figsize=(20, 10))
smooth = savgol_filter(rewards2, window_size2, polyorder)
plt.plot(eposides3, rewards2, label='rewards')
plt.plot(eposides3, smooth, linestyle='--', lw=4, label='smooth of rewards')
plt.ylim(-10, 30)
plt.legend()
plt.xlabel('eposides')
plt.ylabel('evaluate rewards')
# plt.show()
plt.savefig('png_file/dqn_reward_val.png')

win_probs = []
for i in range(0, opp_hps.shape[0], 20):
    win_prob = np.sum(own_hps[i : i + 20] > opp_hps[i : i + 20]) / (20 if i + 20 < opp_hps.shape[0] else opp_hps.shape[0] - i)
    win_probs.append(win_prob)
plt.figure(figsize=(20, 10))
smooth = savgol_filter(win_probs, window_size3, polyorder)
plt.plot(np.arange(len(win_probs)) * 20, win_probs, label='winning probability')
plt.plot(np.arange(len(win_probs)) * 20, smooth, linestyle='--', lw=4, label='smooth of winning probability')
plt.legend()
plt.xlabel('eposides')
plt.ylabel('winning probability during training')
# plt.show()
plt.savefig('png_file/dqn_win_prob_train.png')


win_probs = []
for i in range(0, opp_hps_val.shape[0], 6):
    if i > 275:
        dsj = 0
    win_prob = np.sum(own_hps_val[i : i + 6] > opp_hps_val[i : i + 6]) / (6 if i + 6 < opp_hps_val.shape[0] else opp_hps_val.shape[0] - i)
    win_probs.append(win_prob)
plt.figure(figsize=(20, 10))
smooth = savgol_filter(win_probs, window_size3, polyorder)
plt.plot(np.arange(len(win_probs)) * 6 / 3 * 10, win_probs, label='winning probability')
plt.plot(np.arange(len(win_probs)) * 6 / 3 * 10, smooth, linestyle='--', lw=4, label='smooth of winning probability')
plt.xlabel('eposides')
plt.ylabel('winning probability during evaluating')
plt.legend()
# plt.show()
plt.savefig('png_file/dqn_win_prob_val.png')