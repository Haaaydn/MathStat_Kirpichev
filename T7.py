import math
from scipy.stats import chi2


l = 0.61
m = 1
k = 3
n = 200


def P(k):
    p = (l ** k) / (math.factorial(k)) * math.exp(-l)
    print(f"P{k} = {p}")
    return p


p = [0] * 5
for i in range(5):
    p[i] = P(i)

p_new = [0] * k
p_new[0] = p[0]
p_new[1] = p[1]
p_new[2] = 1 - (p[0] + p[1])

mi = [109, 65, 26]

delta = 0
for i in range(3):
    delta += ((mi[i] - n * p_new[i]) ** 2) / (n * p_new[i])

print(f"delta = {delta}")

df = k - m - 1
pval = chi2.sf(delta, df)

print(f"P(χ²({df}) > {delta}) = {pval}")