import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize

print("--------------------------------------  a) Критерий χ² ------------------------------------------")
n = 100
vals = np.arange(10)
m = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])

thetta1 = 0
thetta2 = float(850 / 93)

p = [0] * 10
p[0] = 0.5 / thetta2 - thetta1
p[9] = (thetta2 - 8.5) / thetta2 - thetta1
for i in range(1, 9):
    p[i] = 1 / thetta2 - thetta1

p = np.array(p)

delta = np.sum((m - n * p) ** 2 / (n * p))
print(f"delta = {delta}")

pval = stats.chi2.sf(delta, 7)

print(f"p-value = {pval:.10f}")


print("--------------------------------  a) Критерий Колмогорова ---------------------------------------")


def F_left(x, xn): return np.sum(xn < x) / len(xn)
def F_right(x, xn): return np.sum(xn <= x) / len(xn)


def sup(xn, F):
    sup = 0
    for x in xn:
        first = abs(F_left(x, xn) - F(x))
        second = abs(F_right(x, xn) - F(x))
        sup = max(sup, first, second)
    return sup


xn = np.repeat(vals, m)

alpha1 = np.mean(xn)
alpha2 = np.mean(xn ** 2)

thetta2 = alpha1 + np.sqrt(4 * alpha1 ** 2 - alpha2)
thetta1 = alpha1 - np.sqrt(4 * alpha1 ** 2 - alpha2)

delta_wave = np.sqrt(n) * sup(xn, lambda x: (x - thetta1) / (thetta2 - thetta1))
print(f"delta_wave = {delta_wave}")

N = 50000
delta = [0] * N
for i in range(N):
    xn_star = np.random.uniform(thetta1, thetta2, n)

    alpha1_star = np.mean(xn_star)
    alpha2_star = np.mean(xn_star ** 2)

    thetta2_star = alpha1_star + np.sqrt(4 * alpha1_star ** 2 - alpha2_star)
    thetta1_star = alpha1_star - np.sqrt(4 * alpha1_star ** 2 - alpha2_star)

    delta[i] = np.sqrt(n) * sup(
        xn,
        lambda x: (x - thetta1_star) / (thetta2_star - thetta1_star)
    )

delta_sort = np.sort(delta)
l = np.sum(delta_sort >= delta_wave)

print(f"l = {l}")
print(f"p-value = l / N = {l / N}")

print("--------------------------------------  б) Критерий χ² ------------------------------------------")


def L(params):
    a, sigma2 = params
    p = [0] * 10
    borders = np.arange(9) + 0.5

    p[0] = stats.norm.cdf(0.5, loc=a, scale=np.sqrt(sigma2))
    p[9] = 1 - stats.norm.cdf(8.5, loc=a, scale=np.sqrt(sigma2))
    for i in range(1, 9):
        p[i] = stats.norm.cdf(
            borders[i], loc=a, scale=np.sqrt(sigma2)
        ) - stats.norm.cdf(
            borders[i - 1], loc=a, scale=np.sqrt(sigma2)
        )

    p = np.array(p)

    return -np.sum(m * np.log(p))


bounds = [(-np.inf, np.inf), (0, np.inf)]

pval = optimize.minimize(
    L,
    [np.mean(xn), np.mean(xn ** 2) - np.mean(xn) ** 2],
    method='L-BFGS-B',
    bounds=bounds
)

# print(result)

# После численного поиска:
a = 4.79
sigma2 = 7.18

p = [0] * 10
borders = np.arange(9) + 0.5

p[0] = stats.norm.cdf(0.5, loc=a, scale=np.sqrt(sigma2))
p[9] = 1 - stats.norm.cdf(8.5, loc=a, scale=np.sqrt(sigma2))
for i in range(1, 9):
    p[i] = stats.norm.cdf(
        borders[i], loc=a, scale=np.sqrt(sigma2)
    ) - stats.norm.cdf(
        borders[i - 1], loc=a, scale=np.sqrt(sigma2)
    )

p = np.array(p)
delta = np.sum((m - n * p) ** 2 / (n * p))

print(f"delta = {delta}")

pval = stats.chi2.sf(delta, 7)

print(f"p-value = {pval:.10f}")

print("--------------------------------  б) Критерий Колмогорова ---------------------------------------")


def sup(xn, F):

    unique_x = np.unique(xn)
    n = len(xn)

    left_counts = np.array([np.sum(xn < x) for x in unique_x])
    right_counts = np.array([np.sum(xn <= x) for x in unique_x])

    left_ecdf = left_counts / n
    right_ecdf = right_counts / n

    F_vals = F(unique_x)

    left_diff = np.abs(left_ecdf - F_vals)
    right_diff = np.abs(right_ecdf - F_vals)

    return np.max(np.maximum(left_diff, right_diff))


thetta1 = alpha1
thetta2 = alpha2 - alpha1 ** 2

delta_wave = np.sqrt(n) * sup(
    xn,
    lambda x: stats.norm.cdf(x, loc=thetta1, scale=np.sqrt(thetta2))
)
print(f"delta_wave = {delta_wave}")

delta = [0] * N
for i in range(N):
    xn_star = np.random.normal(thetta1, np.sqrt(thetta2), n)

    alpha1_star = np.mean(xn_star)
    alpha2_star = np.mean(xn_star ** 2)

    theta1_star = alpha1_star
    theta2_star = alpha2_star - alpha1_star ** 2

    delta[i] = np.sqrt(n) * sup(
        xn_star,
        lambda x: stats.norm.cdf(x, loc=theta1_star, scale=np.sqrt(theta2_star))
    )

l = np.sum(delta >= delta_wave)

print(f"l = {l}")
print(f"p-value = l / N = {l / N}")