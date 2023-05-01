from scipy.signal import lfilter
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import matplotlib

TURN_NUMBER = 200
GAMES = 1000

color = {'single_trend_agent': 'dodgerblue', 'multi_trend_agent': 'springgreen', 'reverse_single_trend_agent': 'indigo',
         'reverse_multi_trend_agent': 'deeppink', 'martingale_agent': 'orange', 'reverse_martingale_agent': 'black',
         'multi_martingale_agent': 'darkslategray', 'reverse_multi_martingale_agent': 'brown',
         'single_trend_agent_zero': 'teal', 'multi_trend_agent_zero': 'crimson',
         'reverse_single_trend_agent_zero': 'peru', 'reverse_multi_trend_agent_zero': 'lightcoral',
         'martingale_agent_zero': 'blueviolet', 'reverse_martingale_agent_zero': 'grey',
         'multi_martingale_agent_zero': 'tomato', 'reverse_multi_martingale_agent_zero': 'olive'}

k_color = list(color.keys())

cash_log = np.load('cash.npy', mmap_mode='r').T
win_log = np.load('win.npy', mmap_mode='r').T
lose_log = np.load('lose.npy', mmap_mode='r').T

n = 1  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

cash_log_mean = []
confidence_interval = []
stds = []
variances = []

cash_log_mean = []
for i in range(16):
    stds_aux = []
    variances_aux = []
    cash_mean = []
    cash_log_confidence_lower = []
    cash_log_confidence_higher = []
    for t in range(TURN_NUMBER):
        total_value = 0
        value_aux = []
        for g in range(GAMES):
            value = cash_log[i][g*TURN_NUMBER + t]
            total_value += value
            value_aux.append(value)
        cash_mean.append(total_value/GAMES)
        confidence_interval_lower, confidence_interval_higher = st.t.interval(confidence=0.95, df=len(value_aux)-1,
                                                                              loc=np.mean(value_aux),
                                                                              scale=st.sem(value_aux))
        stds_aux.append(np.std(value_aux))
        variances_aux.append(np.var(value_aux))
        cash_log_confidence_lower.append(confidence_interval_lower)
        cash_log_confidence_higher.append(confidence_interval_higher)
    stds.append(stds_aux)
    variances.append(variances_aux)
    confidence_interval.append([cash_log_confidence_lower, cash_log_confidence_higher])
    cash_log_mean.append(cash_mean.copy())

stay_agent_list = []
lose_agent_list = []
lose_log_mean = []
win_agent_list = []
win_log_mean = []
for i in range(16):
    win_mean = []
    win_aux = []
    lose_mean = []
    lose_aux = []
    stay_aux = []
    for t in range(TURN_NUMBER):
        total_value_win = 0
        total_value_lose = 0
        total_value_stay = 0
        for g in range(GAMES):
            win_value = win_log[i][g * TURN_NUMBER + t]
            lose_value = lose_log[i][g * TURN_NUMBER + t]
            total_value_win += win_value
            total_value_lose += lose_value
            if not win_value and not lose_value:
                total_value_stay += 1

        win_mean.append(total_value_win / GAMES)
        lose_mean.append(total_value_lose / GAMES)
        win_aux.append(total_value_win)
        lose_aux.append(total_value_lose)
        stay_aux.append(total_value_stay)

    win_log_mean.append(win_mean.copy())
    lose_log_mean.append(lose_mean.copy())
    lose_agent_list.append(sum(lose_aux))
    win_agent_list.append(sum(win_aux))
    stay_agent_list.append(sum(stay_aux))

matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
fig.set_dpi(100)

means = lfilter(b, a, cash_log_mean)
for i, cash_mean in enumerate(means):
    if i == 4:
        break
    cash_mean[0:n-1] = 1000
    confidence_lower, confidence_higher = confidence_interval[i]
    ax.plot(range(TURN_NUMBER), cash_mean, linestyle='-', linewidth=2, label=k_color[i].replace("_", " "), c=color[k_color[i]])
    ax.fill_between(range(TURN_NUMBER), confidence_lower, confidence_higher, alpha=0.1, facecolor=color[k_color[i]])
    print(
        "-------------- {} ---------------\nMean: {}\nStd: {}\nVariance: {}\nConfidence Interval: {} <= p <= {}".format(
            k_color[i].replace("_", " "), cash_mean[-1], stds[i][-1], variances[i][-1], confidence_lower[-1],
            confidence_higher[-1]))
    total_len = win_agent_list[i] + lose_agent_list[i] + stay_agent_list[i]
    print("Win: {} --- Lose: {} --- Stay: {}".format(win_agent_list[i], lose_agent_list[i], stay_agent_list[i]))
    print("Win: {}% --- Lose: {}% --- Stay: {}%".format((win_agent_list[i]/total_len)*100, (lose_agent_list[i]/total_len)*100, (stay_agent_list[i]/total_len)*100))

ax.legend(loc="lower left", mode="expand", ncol=2, prop={'size': 16})
ax.set_xlabel('Turn', fontsize=20)
ax.set_ylabel('Cash', fontsize=20)
ax.set_ylim([550, 1250])
ax.set_xlim([0, TURN_NUMBER])
ax.grid()
plt.savefig("cash.pdf", format="pdf", bbox_inches="tight", backend='pgf')
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
fig.set_dpi(100)

for i, cash_mean in enumerate(means):
    if i < 4:
        continue
    elif i == 8:
        break
    cash_mean[0:n - 1] = 1000
    confidence_lower, confidence_higher = confidence_interval[i]
    ax.plot(range(TURN_NUMBER), cash_mean, linestyle='-', linewidth=2, label=k_color[i].replace("_", " "), c=color[k_color[i]])
    ax.fill_between(range(TURN_NUMBER), confidence_lower, confidence_higher, alpha=0.1, facecolor=color[k_color[i]])
    print(
        "-------------- {} ---------------\nMean: {}\nStd: {}\nVariance: {}\nConfidence Interval: {} <= p <= {}".format(
            k_color[i].replace("_", " "), cash_mean[-1], stds[i][-1], variances[i][-1], confidence_lower[-1],
            confidence_higher[-1]))
    total_len = win_agent_list[i] + lose_agent_list[i] + stay_agent_list[i]
    print("Win: {} --- Lose: {} --- Stay: {}".format(win_agent_list[i], lose_agent_list[i], stay_agent_list[i]))
    print("Win: {}% --- Lose: {}% --- Stay: {}%".format((win_agent_list[i]/total_len)*100, (lose_agent_list[i]/total_len)*100, (stay_agent_list[i]/total_len)*100))

ax.legend(loc="lower left", mode="expand", ncol=2, prop={'size': 16})
ax.set_xlabel('Turn', fontsize=20)
ax.set_ylabel('Cash', fontsize=20)
ax.set_ylim([550, 1250])
ax.set_xlim([0, TURN_NUMBER])
ax.grid()
fig.set_dpi(100)
plt.savefig("cash2.pdf", format="pdf", bbox_inches="tight", backend='pgf')
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
fig.set_dpi(100)

for i, cash_mean in enumerate(means):
    if i < 8:
        continue
    elif i == 12:
        break
    cash_mean[0:n - 1] = 1000
    confidence_lower, confidence_higher = confidence_interval[i]
    ax.plot(range(TURN_NUMBER), cash_mean, linestyle='-', linewidth=2, label=k_color[i].replace("_", " "), c=color[k_color[i]])
    ax.fill_between(range(TURN_NUMBER), confidence_lower, confidence_higher, alpha=0.1, facecolor=color[k_color[i]])
    print(
        "-------------- {} ---------------\nMean: {}\nStd: {}\nVariance: {}\nConfidence Interval: {} <= p <= {}".format(
            k_color[i].replace("_", " "), cash_mean[-1], stds[i][-1], variances[i][-1], confidence_lower[-1],
            confidence_higher[-1]))
    total_len = win_agent_list[i] + lose_agent_list[i] + stay_agent_list[i]
    print("Win: {} --- Lose: {} --- Stay: {}".format(win_agent_list[i], lose_agent_list[i], stay_agent_list[i]))
    print("Win: {}% --- Lose: {}% --- Stay: {}%".format((win_agent_list[i]/total_len)*100, (lose_agent_list[i]/total_len)*100, (stay_agent_list[i]/total_len)*100))

ax.legend(loc="lower left", mode="expand", ncol=2, prop={'size': 16})
ax.set_xlabel('Turn', fontsize=20)
ax.set_ylabel('Cash', fontsize=20)
ax.set_ylim([550, 1250])
ax.set_xlim([0, TURN_NUMBER])
ax.grid()
fig.set_dpi(100)
plt.savefig("cash3.pdf", format="pdf", bbox_inches="tight", backend='pgf')
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
fig.set_dpi(100)

for i, cash_mean in enumerate(means):
    if i < 12:
        continue
    cash_mean[0:n - 1] = 1000
    confidence_lower, confidence_higher = confidence_interval[i]
    ax.plot(range(TURN_NUMBER), cash_mean, linestyle='-', linewidth=2, label=k_color[i].replace("_", " "), c=color[k_color[i]])
    ax.fill_between(range(TURN_NUMBER), confidence_lower, confidence_higher, alpha=0.1, facecolor=color[k_color[i]])
    print(
        "-------------- {} ---------------\nMean: {}\nStd: {}\nVariance: {}\nConfidence Interval: {} <= p <= {}".format(
            k_color[i].replace("_", " "), cash_mean[-1], stds[i][-1], variances[i][-1], confidence_lower[-1],
            confidence_higher[-1]))
    total_len = win_agent_list[i] + lose_agent_list[i] + stay_agent_list[i]
    print("Win: {} --- Lose: {} --- Stay: {}".format(win_agent_list[i], lose_agent_list[i], stay_agent_list[i]))
    print("Win: {}% --- Lose: {}% --- Stay: {}%".format((win_agent_list[i]/total_len)*100, (lose_agent_list[i]/total_len)*100, (stay_agent_list[i]/total_len)*100))

ax.legend(loc="lower left", mode="expand", ncol=2, prop={'size': 14})
ax.set_xlabel('Turn', fontsize=20)
ax.set_ylabel('Cash', fontsize=20)
ax.set_ylim([550, 1250])
ax.set_xlim([0, TURN_NUMBER])
ax.grid()
fig.set_dpi(100)
plt.savefig("cash4.pdf", format="pdf", bbox_inches="tight", backend='pgf')
plt.show()
