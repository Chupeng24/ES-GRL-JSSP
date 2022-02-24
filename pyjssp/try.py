import math

import numpy as np
# mu, sigma = 100, 2      # 均值和标准差
mu = np.array([10,100])
sigma = 1
s = np.random.normal(mu, sigma, 2)
print(s)
for val in s:
    z = math.ceil(val)
    print(z)
# print(abs(mu - np.mean(s)) < 1)
# print(abs(sigma - np.std(s, ddof=1)) < 0.01)

