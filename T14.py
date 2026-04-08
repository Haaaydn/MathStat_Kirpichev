from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Параметры
alpha = 0.05

# Квантиль порядка 1 - alpha
quantile = norm.ppf(1 - alpha)

print(f"Квантиль порядка {1-alpha} для N(0, 1) U[1-α]{quantile:.4f}")

x = np.array([-1.11, -6.1, 2.42])
y = np.array([-2.29, -2.91])

x_ = np.mean(x)
y_ = np.mean(y)

delta = (x_ - y_) / np.sqrt(7/6)

print(f'x̄ = {x_}\nȳ = {y_}\ndelta_wave = {delta}')


def power(tetha):
    return 1 - norm.cdf(1.645 - tetha / np.sqrt(7/6))


theta_values = np.linspace(0, 10, 500)

# Вычисление мощности
power_values = [power(theta) for theta in theta_values]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(theta_values, power_values, 'r-', linewidth=2, label=f'W(θ)')

# Настройка графика
plt.xlabel('Разность мат.ожиданий θ = a - b', fontsize=14)
plt.ylabel('Мощность критерия W(θ)', fontsize=14)
plt.title(f'График мощности критерия на заданных выборках', fontsize=16)
plt.grid(True, alpha=1)

# Отметим характерные точки
plt.axhline(y=0.05, color='b', linestyle='--', alpha=0.5, label='α = 0.05')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='θ = 0 (H0: a = b)')

plt.legend()
plt.tight_layout()
plt.savefig("./T14_power_of_criterion_boring.png")