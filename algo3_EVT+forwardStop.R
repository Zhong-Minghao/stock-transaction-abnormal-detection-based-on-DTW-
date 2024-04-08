library(eva)
library(xlsx)

# 设置初始目录
setwd("D:/dfs")

data = read.csv("train+002369.csv",header=TRUE)
vec = as.vector(t(data))
vec = sort(vec)
qt7 = as.numeric(quantile(vec,probs=0.7))
qt95 = as.numeric(quantile(vec,probs=0.95))

u = qt7
beta = 0.1
k = 0
sum_lp = 0
bestFS = 0
best_u = 0
best_theta = c(0,0)

for (i in 1:round(qt95-qt7)) {
  this_vec = vec - u
  this_vec = this_vec[this_vec > 0]
  ad = gpdAd(this_vec)
  p = ad$p.value
  k = k + 1
  sum_lp = sum_lp + log(1 - p)
  FS = -sum_lp / k
  if ((FS < beta)&(FS > bestFS)) {
    bestFS = FS
    best_u = u
    best_theta = ad$theta
  }
  u = u + 1
}

print(best_u)

a = 0.95
this_vec = vec - best_u
this_vec = this_vec[this_vec > 0]
scale = as.numeric(best_theta[1])
shape = as.numeric(best_theta[2])

Qa = best_u + scale/shape*(((length(vec)/length(this_vec)*(1-a))^(-shape))-1)
tau = (Qa + scale + shape*(Qa-best_u))/(1-shape)

print(tau)


# 初始化结果向量
# result <- matrix(0, nrow = 10000, ncol = 4)

# 将statistic,p.value,Scale,Shape储存起来
# result[i,] <- c(ad$statistic, ad$p.value, ad$theta["Scale"], ad$theta["Shape"])
# 将结果输出到excel
# write.xlsx(result, file = "D:/Rresult/result02.xlsx", row.names = TRUE)
