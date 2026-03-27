import numpy as np
from scipy.stats import chi2


m = np.array([
    [33, 43, 80, 144],
    [39, 35, 72, 154]
])

p = [0] * 4
n = 600
n1 = n2 = 300

for i in range(4):
    p[i] = (m[0][i] + m[1][i]) / n

delta = 0

for j in range(2):
    for i in range(4):
        delta += ((m[j][i] - n1 * p[i]) ** 2) / (n1 * p[i])


print(delta)


result = chi2.sf(delta, 3)
print(f"{result:.10f}")