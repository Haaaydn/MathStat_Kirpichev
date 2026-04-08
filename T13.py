from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np


n = 139   # dfn = n-1
s1_length = 5.722
s1_width = 4.612

m = 1000  # dfd = m-1
s2_length = 6.161
s2_width = 5.055

x_length = s1_length ** 2 / s2_length ** 2
x_width = s1_width ** 2 / s2_width ** 2

alpha = 0.05

# Вычисление интеграла от 0 до x
F_left = f.ppf(alpha / 2, n-1, m-1)
F_right = f.ppf(1 - alpha / 2, n-1, m-1)

print(f'F[α/2](n-1, m-1) = {F_left} \nF[1-α/2](n-1, m-1) = {F_right}')

print(f'delta_wave_skull_length = {x_length}')
print(f'delta_wave_skull_width = {x_width}')


def power(tetha):
    return f.cdf(0.77 / tetha, n-1, m-1) + 1 - (f.cdf(1.27 / tetha, n-1, m-1))


theta_values = np.linspace(0.1, 3, 500)

# Вычисление мощности
power_values = [power(theta) for theta in theta_values]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(theta_values, power_values, 'r-', linewidth=2, label=f'W(θ)')

# Настройка графика
plt.xlabel('Отношение дисперсий θ = a²/b²', fontsize=14)
plt.ylabel('Мощность критерия W(θ)', fontsize=14)
plt.title(f'График мощности критерия для черепов (при n={n}, m={m})', fontsize=16)
plt.grid(True, alpha=1)
plt.legend()

# Отметим характерные точки
plt.axhline(y=0.05, color='b', linestyle='--', alpha=0.5, label='α = 0.05')
plt.axvline(x=1.0, color='g', linestyle='--', alpha=0.5, label='θ = 1 (H0: a² = b²)')

plt.legend()
plt.tight_layout()
plt.savefig("./T13_power_of_criterion_skulls.png")