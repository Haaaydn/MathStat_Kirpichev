import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import comb


def print_header(s: str):
    print("\n" + "-" * 60)
    print(s)
    print("-" * 60)


np.random.seed(7)
n = 25
# Экспоненциальное распределение с параметром rate=1 (масштаб=1)
# В numpy экспоненциальное распределение задается через масштаб (scale = 1/λ)
# При p(x) = e^{-x}, λ=1, scale=1
selection = np.random.exponential(scale=1, size=n)

print_header("Т2")
print(f"\nСгенерированная выборка (n={n}):")
print(selection)

# ============================================================================
# a) Определить моду, медиану, размах, оценку коэффициента асимметрии
# ============================================================================

print_header("a) - Описательные статистики")

# Медиана
median = np.median(selection)
print(f"Медиана: {median:.4f}")

mode_result = stats.mode(selection)
if mode_result.count > 1:
    print("Мода:", mode_result.mode)
    print("Количество вхождений моды:", mode_result.count)
else:
    print("Моды нет (все элементы являются модой)")

# Размах
scope = np.max(selection) - np.min(selection)
print(f"Минимум: {np.min(selection):.4f}, Максимум: {np.max(selection):.4f}")
print(f"Размах: {scope:.4f}")

# Коэффициент асимметрии
skewness = stats.skew(selection, bias=False)
print(f"Коэффициент асимметрии (несмещенная оценка): {skewness:.4f}")

# ============================================================================
# b) Построить эмпирическую функцию распределения, гистограмму и boxplot
# ============================================================================

print_header("b) - Построение графиков")

fig, axes = plt.subplots(4, 1, figsize=(16, 9))

ax0 = axes[0]
ax0.set_xlim(-0.5, 4)
ax0.scatter(selection, [np.exp(-selection)])
ax0.set_title('b) Наша выборка')

# 1. Эмпирическая функция распределения
ax1 = axes[1]
x_sorted = np.sort(selection)
y_exp = np.arange(1, n + 1) / n
ax1.step(x_sorted, y_exp, where='post', linewidth=2, label='Эмпирич. ф-ция распр.')

# Добавляем теоретическую функцию распределения Exp(1)
x_theor = np.linspace(0, np.max(selection) * 1.1, 100)
y_theor = stats.expon.cdf(x_theor, scale=1)
ax1.plot(x_theor, y_theor, 'r--', label='Теоретич. ф-ция распр.(Exp(1))')

ax1.set_xlabel('x')
ax1.set_ylabel('F(x)')
ax1.set_title('Эмпирическая функция распределения')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 4)

# 2. Гистограмма
ax2 = axes[2]
ax2.hist(selection, bins='auto', density=True, alpha=0.7,
         color='steelblue', edgecolor='black', label='Гистограмма')

# Добавляем теоретическую плотность Exp(1)
x_pdf = np.linspace(0, np.max(selection) * 1.1, 100)
y_pdf = stats.expon.pdf(x_pdf, scale=1)
ax2.plot(x_pdf, y_pdf, 'r-', label='Теоретическая плотность (Exp(1))')

ax2.set_xlabel('x')
ax2.set_ylabel('Плотность')
ax2.set_title('Гистограмма и теоретическая плотность')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.5, 4)

# 3. Boxplot
ax3 = axes[3]
ax3.boxplot(selection, vert=False, patch_artist=True, boxprops=dict(facecolor="steelblue"))
ax3.set_xlabel('Значения')
ax3.set_title('Boxplot')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.5, 4)
ax3.set_yticklabels(['Выборка'])

plt.tight_layout()
plt.savefig('T2_b.png', dpi=150, bbox_inches='tight')

# ============================================================================
# c) Сравнить оценку плотности среднего арифметического по ЦПТ с бутстрапом
# ============================================================================

print_header("c) - Сравнение оценки плотности среднего по ЦПТ и бутстрапу")

# Теоретически: по ЦПТ среднее арифметическое ~ N(μ, σ²/n)
# Для Exp(1): μ = 1, σ² = 1
mu_theor = 1
sigma_theor = 1
se_theor = sigma_theor / np.sqrt(n)

print("Теоретические параметры для Exp(1):")
print(f"   |-Мат. ожидание = {mu_theor}")
print(f"   |-Дисперсия = {sigma_theor**2}")
print(f"   |-Стандартная ошибка среднего = {se_theor:.4f}")

sample_mean = np.mean(selection)
print(f"\nСреднее нашей выборки: {sample_mean:.4f}")

# Бутстрап для распределения среднего
NB = 10000  # Cделал побольше, потому что так точнее и не сильно дольше
bootstrap_means = np.zeros(NB)

# Генерируем бутстрап-выборку (с возвращением)
for i in range(NB):
    bootstrap_selection = np.random.choice(selection, size=n, replace=True)
    bootstrap_means[i] = np.mean(bootstrap_selection)

bootstrap_mean_mean = np.mean(bootstrap_means)
bootstrap_mean_std = np.std(bootstrap_means, ddof=1)

print("\nБутстрап-оценки распределения среднего:")
print(f"   |-Среднее бутстрап-средних: {bootstrap_mean_mean:.4f}")
print(f"   |-Стандартное отклонение бутстрап-средних: {bootstrap_mean_std:.4f}")
print(f"   |-Отличие от теоретического: {abs(bootstrap_mean_std - se_theor):.4f}")

fig, ax1 = plt.subplots(1, figsize=(14, 5))

ax1.hist(bootstrap_means, bins=30, density=True, alpha=0.7,
         color='steelblue', edgecolor='black', label='Бутстрап-средние')

# Нормальное приближение по ЦПТ
x_norm = np.linspace(np.min(bootstrap_means), np.max(bootstrap_means), 100)
y_norm = stats.norm.pdf(x_norm, loc=mu_theor, scale=se_theor)
ax1.plot(x_norm, y_norm, 'r-', linewidth=2, label='Нормальное приближение (ЦПТ)')

ax1.set_xlabel('Среднее арифметическое')
ax1.set_ylabel('Плотность')
ax1.set_title('с) Бутстрап и ЦПТ')
ax1.legend()
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('T2_c.png', dpi=150, bbox_inches='tight')

# ============================================================================
# d) Бутстраповская оценка плотности коэффициента асимметрии
# ============================================================================

print_header("d) - Бутстрап для коэффициента асимметрии γ ")

# Бутстрап для коэффициента асимметрии
bootstrap_skewness = np.zeros(NB)

for i in range(NB):
    bootstrap_sample = np.random.choice(selection, size=n, replace=True)
    bootstrap_skewness[i] = stats.skew(bootstrap_sample, bias=False)

skewness_mean = np.mean(bootstrap_skewness)
skewness_std = np.std(bootstrap_skewness, ddof=1)

print(f"Коэффициент асимметрии исходной выборки: {skewness:.4f}")
print("\nБутстрап-оценки для коэффициента асимметрии:")
print(f"   |-Среднее бутстрап-значений: {skewness_mean:.4f}")
print(f"   |-Стандартное отклонение: {skewness_std:.4f}")


# Оценим вероятность P(коэффициент асимметрии < 1)
prob_skewness_less_1 = np.mean(bootstrap_skewness < 1)
print(f"\nОценка вероятности P(γ < 1): {prob_skewness_less_1:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bootstrap_skewness, bins=30, density=True, alpha=0.7,
        color='steelblue', edgecolor='black', label='Бутстрап-коэффициенты асимметрии')

ax.axvline(x=1, color='red', linewidth=2, linestyle='--', label='γ = 1')
ax.axvline(x=skewness, color='green', linewidth=2, linestyle='-',
           label=f'Исходное значение = {skewness:.2f}')

ax.set_xlabel('Коэффициент асимметрии')
ax.set_ylabel('Плотность')
ax.set_title('d) Бутстрап-распределение коэффициента асимметрии')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('T2_d.png', dpi=150, bbox_inches='tight')

# ============================================================================
# e) Сравнить плотность распределения медианы с бутстраповской оценкой
# ============================================================================

print_header("e) - Сравнение плотности медианы по теории и бутстрапу")

# Бутстрап для медианы
bootstrap_median = [np.median(np.random.choice(selection, n, True)) for _ in range(NB)]


def exact_median_pdf(x):
    """Точная плотность распределения медианы (X_(13) для n=25)"""
    k = 13  # позиция медианы для n=25
    C = comb(n, k-1) * k
    F = 1 - np.exp(-x)
    f = np.exp(-x)
    return C * (F**(k-1)) * ((1-F)**(n-k)) * f


# Асимптотическое приближение
true_median = np.log(2)  # медиана Exp(1)
f_me = stats.expon.pdf(true_median)  # плотность в точке медианы = 0.5
asymptotic_se = 1/(2*np.sqrt(n)*f_me)  # = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bootstrap_median, bins=30, density=True, alpha=0.7, 
        color='steelblue', edgecolor='black', label='Бутстрап')

x_vals = np.linspace(0.2, 1.5, 200)
ax.plot(x_vals, stats.norm.pdf(x_vals, true_median, asymptotic_se), 
        'r-', linewidth=2, label='Асимптотика (нормальная)')
ax.plot(x_vals, exact_median_pdf(x_vals), 
        'g--', linewidth=2, label='Точное распределение')
ax.axvline(true_median, color='purple', linestyle=':', label=f'Истинная медиана={true_median:.3f}')
ax.set_xlabel('Медиана')
ax.set_ylabel('Плотность')
ax.set_title(f'e) Распределение медианы (n={n})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('T2_e.png', dpi=150, bbox_inches='tight')
# Сравнение стандартных отклонений
print("Стандартные отклонения медианы:")
print(f"  Бутстрап: {np.std(bootstrap_median):.4f}")
print(f"  Асимптотика: {asymptotic_se:.4f}")
