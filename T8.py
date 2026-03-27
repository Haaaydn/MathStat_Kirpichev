import numpy as np
from scipy.stats import chi2


m = np.array([
    [25, 50, 25],
    [52, 41, 7]
])

p = np.array([1/2, 1/2])
q = np.array([77/200, 91/200, 32/200])
n = 200

delta = 0

for i in range(2):
    for j in range(3):
        delta += ((m[i][j] - n * p[i] * q[j]) ** 2) / (n * p[i] * q[j])

print(delta)

pval = chi2.sf(delta, 2)

print(f"{pval:.10f}")