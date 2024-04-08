import numpy as np
import pandas as pd
from scipy.stats import genpareto, anderson_ksamp, linregress
import matplotlib.pyplot as plt
import sys
import warnings
# from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


class ProgressBar:
    """This progress bar was taken from PYMC
    """

    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.percent_done_sign = -1
        self.width = 40
        self.__update_amount(0)
        self.animate = self.animate_ipython

    def __str__(self):
        return str(self.prog_bar)

    def animate_ipython(self, iter):
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        if percent_done == self.percent_done_sign:
            pass
        else:
            all_full = self.width - 2
            num_hashes = int(round((percent_done / 100.0) * all_full))
            self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
            pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
            pct_string = '%d%%' % percent_done
            self.prog_bar = self.prog_bar[0:pct_place] + (pct_string + self.prog_bar[pct_place + len(pct_string):])
            print(str(self.prog_bar))
            self.percent_done_sign = percent_done


# 设置模拟参数
num_simulations = 5000  # 模拟次数
dot_simulations = 500
conf_level = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]  # 置信度列表
size = 100  # 样本大小
c = -0.5  # 形状参数（xi）
scale = 5  # 尺度参数(sigma)

p_value_df = pd.DataFrame(columns=['statics', 'p_value'])
statistic = lambda x: anderson_ksamp([x, sample0])

p = ProgressBar(dot_simulations * num_simulations)
dm_count = 0

for i in range(dot_simulations):
    sample0 = genpareto(c=c, scale=scale).rvs(size)
    sample = genpareto(c=c, scale=scale).rvs(size)
    obs = statistic(sample)

    # 重复上述过程10000次以获取检验统计量值的分布
    test_statistics = np.zeros(num_simulations)
    for j in range(num_simulations):
        sample = genpareto(c=c, scale=scale).rvs(size)
        res = statistic(sample)
        test_statistics[j] = res.statistic

        dm_count += 1

    p.animate(dm_count)

    p_value = np.mean(test_statistics > obs.statistic)

    p_value_df.loc[len(p_value_df.index)] = [obs.statistic, p_value]

p_value_df.sort_values(by=['p_value'], inplace=True)

# plt.scatter(p_value_df['p_value'], p_value_df['statics'])
plt.scatter(p_value_df['p_value'], np.log(p_value_df['statics']+1))
plt.show()

# 手动剔除极端值
p_value_df['log_sta'] = np.log(p_value_df['statics']+1)
p_value_df = p_value_df.loc[p_value_df['log_sta'] > -3].copy()

p_value_df.dropna(axis=0, how='any', inplace=True)

slope, intercept, r_value, p_value, std_err = linregress(p_value_df['p_value'], p_value_df['log_sta'])

print(slope, intercept, r_value, p_value, std_err)

this_res = pd.DataFrame([[c, slope, intercept, r_value, p_value, std_err]], columns=['xi', 'slope', 'intercept', 'r_value', 'p_value', 'std_err'])

this_res.to_csv(f'linregress+xi={c}.csv')

# 设置上下限
# gumbel_r.a = 0
# gumbel_r.b = 20

