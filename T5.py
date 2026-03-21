import numpy as np

thetta = 3
n = 100
xn = thetta + thetta * np.random.random(n)

thetta_1 = 2/3 * np.mean(xn)


def current_method():
    x_max = np.max(xn)
    t1 = 1 + 0.025 ** (1 / n)
    t2 = 1 + 0.975 ** (1 / n)
    left = x_max / t2
    right = x_max / t1
    length = right - left
    print(f'Точный метод: ({left}, {right}),\n длина интервала: {length} \n')
    return length


def assimpt_method():
    alpha_1 = np.mean(xn)
    alpha_2 = np.mean(xn ** 2)
    t1 = -1.96
    t2 = 1.96
    left = thetta_1 - 2/3 * t2 * np.sqrt((alpha_2 - alpha_1 ** 2) / n)
    right = thetta_1 - 2/3 * t1 * np.sqrt((alpha_2 - alpha_1 ** 2) / n)
    length = right - left
    print(f'Ассимптотический метод (ОММ ~θ1 = 2/3*xср): ({left}, {right}),\n длина интервала: {length} \n')
    return length


def bootstrap_method():
    samples = np.random.choice(xn, (1000, n), replace=True)
    var_series = np.sort(np.apply_along_axis(lambda x: 2/3 * np.mean(x) - thetta_1, 1, samples))
    k1, k2 = 25, 975
    left = thetta_1 - var_series[k2]
    right = thetta_1 - var_series[k1]
    length = right - left
    print(f'Непараметрический Bootstrap: ({left}, {right}),\n длина интервала: {length} \n')
    return length


l_curr = current_method()
l_assimpt = assimpt_method()
l_boot = bootstrap_method()

best = 'Точный метод'  
if l_assimpt < l_curr and l_assimpt < l_boot:
    best = 'Ассимптотический метод'  
if l_boot < l_assimpt and l_boot < l_curr:
    best = 'Непараметрический Bootstrap'  
print(f'Лучший метод для θ={thetta} оказался {best}')