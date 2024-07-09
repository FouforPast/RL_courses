import re
import numpy as np

name = 'eval_log_valid.txt'
text = open(name, 'r').read()

info = []
# text = """
# At the end, own_hp 0: opp_hp 215. you lose.
# Episode 0: loss is 0.582, rewards is -18.89
# """
# 提取own_hp和opp_hp的数字
val = re.compile(r"At the end, own_hp (?P<own_hp>\d+): opp_hp (?P<opp_hp>\d+)\.")
for i in val.finditer(text):
    info.append((i.group('own_hp'), i.group('opp_hp')))



own_hps = np.array([int(_[0]) for _ in info])
opp_hps = np.array([int(_[1]) for _ in info])


import matplotlib.pyplot as plt
window_size = 100
polyorder = 4
plt.figure(figsize=(20, 10))
# plt.subplot(2, 2, 1)
# plt.plot(np.arange(own_hps.shape[0]), own_hps - opp_hps, label='the difference of hp')
success_rate = np.sum(own_hps > opp_hps) / own_hps.shape[0]
plt.hist(own_hps - opp_hps, bins=len(own_hps) // 4, color='steelblue', edgecolor='white', label='the difference of hp, winning probability: {:.2f}%'.format(success_rate * 100))
# smooth = savgol_filter(own_hps - opp_hps, window_size, polyorder)
# plt.plot(np.arange(own_hps.shape[0]), smooth, linestyle='--', label='smooth of the difference of hp')
plt.xlabel('the difference of hp')
plt.ylabel('count')
plt.legend()
# plt.show()
plt.savefig('png_file/eval.png')