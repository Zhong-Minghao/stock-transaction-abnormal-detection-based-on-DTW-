import sympy as sp
import numpy as np
import warnings
from scipy.stats import genpareto


# 生成超越量y
y = np.random.randn(15)+2
# x,p,z= sp.symbols('x p z',positive=True)#x.p.z都是正数，
xi, sigma = sp.symbols('xi sigma')
L = -len(y) * sp.log(sigma) - (1 + 1/xi) * sum([sp.log(1+xi*i/sigma) for i in y])  # 似然函数
print(L)

# xi等于0的情况
# L = -len(y) * sp.log(sigma) - (1/sigma) * np.sum(y)

eq_1 = L.diff(xi)
eq_2 = L.diff(sigma)
for i in range(len(y)):
    exec(f'eq{i} = 1+xi*{y[i].item()}/sigma')
sol = sp.nsolve((eq_1, eq_2), (xi, sigma), (0.1, 0.1), verify=False)  # [(exec(f'eq{i}>0') for i in range(len(xs)))]

if (len(sol) == 2) & (sol[0] < 1) & (sol[1] > 0):
    xi_bar, sigma_bar = sol[0], sol[1]
else:
    xi_bar, sigma_bar = 1, 1
    warnings.warn('出现多个解！')
    raise Exception

# 验证成立条件
if np.all(y*xi_bar/sigma_bar+1 > 0):
    print('log括号条件成立')
else:
    warnings.warn('log括号条件不成立')

# loc, scale, shapes = genpareto.fit(y)

n = y.shape[0]
y.sort()
zs = 1-(1+xi_bar*y/sigma_bar)**(-1/xi_bar)
zs = zs.astype('float')
log_zs = np.log(zs)
log_zs_1 = np.log(1-zs[::-1])
log_zs_sum = log_zs + log_zs_1
coef = np.linspace(1, 2*n-1, n, endpoint=True)

A2 = - n - (1/n) * np.sum(coef * log_zs_sum)

# A2大于表中对应的数，就拒绝原假设


# import numpy as np
# from scipy.stats import genpareto
#
# # 设置阈值、尺度参数和形状参数
# u, sigma, xi = 5, 2, 0.5
#
# # 生成一个广义 Pareto 分布的样本
# rvs = genpareto.rvs(xi, loc=u, scale=sigma, size=100)
#
# # 计算每个样本对应的分布函数值
# F_values = 1 - (1 + xi * (rvs - u) / sigma)**(-1/xi)
#
# # 对分布函数值进行排序，并转换为 z 值
# ranked_F_values = np.argsort(np.argsort(F_values))
# z_values = (ranked_F_values + 1) / (len(ranked_F_values) + 1)
#
# # 输出转换后的 z 值
# print(z_values)
