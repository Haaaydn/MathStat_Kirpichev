import numpy as np

thetta = 3
n = 100
xn = xn = np.random.pareto(a=thetta-1, size=n) + 1
thetta_wave = 1 + n / (np.sum(np.log(xn)))


def med(t):
    return 2 ** (1 / (t - 1))


def sigma(t):
    return np.log(2) * med(t) / (t - 1)


def I(t):
    return 1 / (t - 1) ** 2


def assimpt_method_med():
    t1 = -1.96
    t2 = 1.96
    left = med(thetta_wave) - t2 * sigma(thetta_wave) / np.sqrt(n)
    right = med(thetta_wave) - t1 * sigma(thetta_wave) / np.sqrt(n)
    length = right - left
    print(f'Ассимптотический метод медиана: ({left}, {right}),\n длина интервала: {length} \n')
    return length


def assimpt_method_thetta():
    t1 = -1.96
    t2 = 1.96
    left = thetta_wave - t2 / np.sqrt(n * I(thetta_wave))
    right = thetta_wave - t1 / np.sqrt(n * I(thetta_wave))
    length = right - left
    print(f'Ассимптотический метод thetta: ({left}, {right}),\n длина интервала: {length} \n')
    return length


def bootstrap_not_param():
    samples = np.random.choice(xn, (1000, n), replace=True)
    var_series = np.sort(
        np.apply_along_axis(
            lambda x: 1 + n / (np.sum(np.log(x))) - thetta_wave,
            axis=1,
            arr=samples
        )
    )
    k1, k2 = 25, 975
    left = thetta_wave - var_series[k2]
    right = thetta_wave - var_series[k1]
    length = right - left
    print(f'Непараметрический Bootstrap: ({left}, {right}),\n длина интервала: {length} \n')
    return length


def bootstrap_param():
    boostrap_samples = np.random.pareto(a=thetta_wave-1, size=(50000, n)) + 1
    var_series = np.sort(
        np.apply_along_axis(
            lambda x: 1 + n / (np.sum(np.log(x))) - thetta_wave,
            axis=1,
            arr=boostrap_samples)
        )
    k1, k2 = 1250, 48750
    left = thetta_wave - var_series[k2]
    right = thetta_wave - var_series[k1]
    length = right - left
    print(f'Параметрический Bootstrap: ({(left)}, {(right)}),\n длина интервала: {length} \n')
    return length


l_assimpt_med = assimpt_method_med()
l_assimpt_thetta = assimpt_method_thetta()
l_boot_not_param = bootstrap_not_param()
l_boot_param = bootstrap_param()