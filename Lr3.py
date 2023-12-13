import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, interp1d, UnivariateSpline
from statistics import mean
from scipy.integrate import quad
import math


# Сделал
def number_1():
    x = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.2]
    y = [-3.5, -4.8, -2.1, 0.2, 0.9, 2.3, 3.7]

    k_0 = np.polyfit(x, y, 0)
    y_appr_0 = [k_0 for _ in x]
    k_1 = np.polyfit(x, y, 1)
    y_appr_1 = [i*k_1[0] + k_1[1] for i in x]
    k_2 = np.polyfit(x, y, 2)
    y_appr_2 = [i*i*k_2[0] + i*k_2[1] + k_2[2] for i in x]
    k_3 = np.polyfit(x, y, 3)
    y_appr_3 = [(i**3)*k_3[0] + i*i*k_3[1] + i*k_3[2] + k_3[3] for i in x]
    k_4 = np.polyfit(x, y, 4)
    y_appr_4 = [(i**4)*k_4[0] + (i**3)*k_4[1] + i*i*k_4[2] + i*k_4[3] + k_4[4] for i in x]
    k_5 = np.polyfit(x, y, 5)
    y_appr_5 = [(i**5)*k_5[0] + (i**4)*k_5[1] + (i**3)*k_5[2] + i*i*k_5[3] + i*k_5[4] + k_5[5] for i in x]
    k_6 = np.polyfit(x, y, 6)
    y_appr_6 = [(i**6)*k_6[0] + (i**5)*k_6[1] + (i**4)*k_6[2] + (i**3)*k_6[3] + i*i*k_6[4] + i*k_6[5] + k_6[6] for i in x]

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='black')
    ax.plot(x, y_appr_0, label='deg = 0')
    ax.plot(x, y_appr_1, label='deg = 1')
    ax.plot(x, y_appr_2, label='deg = 2')
    ax.plot(x, y_appr_3, label='deg = 3')
    ax.plot(x, y_appr_4, label='deg = 4')
    ax.plot(x, y_appr_5, label='deg = 5')
    ax.plot(x, y_appr_6, label='deg = 6')
    ax.legend()
    plt.show()

    print(f'Погрешности\n'
          f'1: {mean([abs(y[i] - y_appr_1[i]) for i in range(len(y))])}\n'
          f'2: {mean([abs(y[i] - y_appr_2[i]) for i in range(len(y))])}\n'
          f'3: {mean([abs(y[i] - y_appr_3[i]) for i in range(len(y))])}\n'
          f'4: {mean([abs(y[i] - y_appr_4[i]) for i in range(len(y))])}\n'
          f'5: {mean([abs(y[i] - y_appr_5[i]) for i in range(len(y))])}\n'
          f'6: {mean([abs(y[i] - y_appr_6[i]) for i in range(len(y))])}')


# number_1()


# Сделал
def number_2():
    x = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    y = [5.17, 7.78, 11.14, 15.09, 19.42, 23.11, 26.25, 28.6, 30.3]
    interp_x = [3.75, 4.75, 5.25]
    Y0 = splrep(x, y)
    interp_y = []
    for i in interp_x:
        interp_y.append(splev(i, Y0))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(interp_x, interp_y)
    ax.scatter(interp_x, interp_y)
    plt.show()
    print(interp_y)

# number_2()


# Сделал
def number_3():
    def f(x):
        return 2 * (np.cos(x) + x)

    # Точки для интерполяции
    xinter = np.array(
        [0, 3.14 / 6, 3.14 / 3, 3.14 / 2, (2 * 3.14) / 3, (5 * 3.14) / 6, 3.14, (7 * 3.14) / 6, (4 * 3.14) / 3,
         (3 * 3.14) / 2, (5 * 3.14) / 3, (11 * 3.14) / 6, 2 * 3.14])
    yinter = f(xinter)
    # Сетка для вычисления погрешности
    x_grid = np.linspace(0, 2 * 3.14, 100)
    ytrue = f(x_grid)
    fstepwise = interp1d(xinter, yinter, kind='zero')
    ystepwise = fstepwise(x_grid)
    flinear = interp1d(xinter, yinter, kind='linear')
    ylinear = flinear(x_grid)
    fcubic = interp1d(xinter, yinter, kind='cubic')
    ycubic = fcubic(x_grid)
    yspline = UnivariateSpline(xinter, yinter)(x_grid)
    # Построение графиков
    plt.style.use("bmh")
    plt.plot(x_grid, ytrue, label='function')
    plt.plot(x_grid, ystepwise, label='stepwise-inter')
    plt.plot(x_grid, ylinear, label='linear-inter')
    plt.plot(x_grid, ycubic, label='cubic-inter')
    plt.plot(x_grid, yspline, label='spline-inter')
    plt.plot(xinter, yinter, '*', color="blue", label='interpolation points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('interpolation')
    plt.grid(True)
    plt.show()
    # Погрешности
    errorstepwise = np.abs(ystepwise - ytrue)
    errorlinear = np.abs(ylinear - ytrue)
    errorcubic = np.abs(ycubic - ytrue)
    errorspline = np.abs(yspline - ytrue)
    print("Погрешности интерполяции:")
    print("stepwise-inter:", np.max(errorstepwise))
    print("linear-inter:", np.max(errorlinear))
    print("cubic-inter:", np.max(errorcubic))
    print("spline-inter:", np.max(errorspline))


# number_3()


# Сделал
def number_4():
    x, a, b = sympy.symbols('x, a, b')

    def f(x):
        return (x / (a + b * x**2))

    print(sympy.integrate(f(x), (x, 1, 3)))


# number_4()


# Сделал
def number_5():
    x = sympy.Symbol('x')

    def integrand(x):
        return ((sympy.log(x) + sympy.exp(x)) / 3 * x**2)

    print(sympy.integrate(integrand(x), x))


# number_5()


# Сделал
def number_6():
    from scipy.integrate import quad
    from sympy import symbols, exp, sin, integrate, lambdify
    from scipy import integrate as scipy_integrate

    x = symbols('x')
    y = exp(x) / (1 + sin(3 * x))
    a1 = 0.5
    a0 = 0.1

    # sympy
    def f(x):
        return exp(x) / (1 + sin(3 * x))

    res = quad(f, a0, a1)[0]

    print('Значение интеграла через sympy:', res)

    # Метод трапеции
    n_trap = 10  # количество интервалов
    dx_trap = (a1 - a0) / n_trap  # шаг разбиения отрезка интегрирования
    x_trap = [a0 + i * dx_trap for i in range(n_trap)]  # список точек разбиения отрезка интегрирования
    y_trap = [y.evalf(subs={x: xi}) for xi in x_trap]  # для каждой точки вычисляется значение
    trap_int = scipy_integrate.trapz(y_trap, x_trap)  # integrate.trapz из scipy интегрирует методом трапеции
    print('Метод трапеции:', trap_int)

    # Метод Симпсона
    n_simps = 10
    dx_simps = (a1 - a0) / n_simps
    x_simps = [a0 + i * dx_simps for i in range(n_simps)]
    y_simps = [y.evalf(subs={x: xi}) for xi in x_simps]
    simps_int = scipy_integrate.simps(y_simps, x_simps)
    print('Метод Симпсона:', simps_int)

    # Метод Ньютона-Котса
    y_fn = lambdify(x, y)
    n_quad = 8
    quad_int = scipy_integrate.fixed_quad(y_fn, a0, a1, n=8)[0]
    print('Метод Ньютона-Котса 8 порядка:', quad_int)

# number_6()


# Сделал
def number_7():
    x = sympy.Symbol('x')
    print(sympy.solve(2**x + 5*x - 3, x))


# number_7()


# Сделал
def number_8():
    A = np.array([[4, 1, -1], [1, -1, 1], [2, -3, -3]])
    B = np.array([[6], [4], [4]])
    A_B = np.concatenate((A, B), axis=1)
    print(A_B)
    if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(A_B):
        print('Система линейных уравнений совместна')

        opred_A = np.linalg.det(A)
        x_i = []
        # print(f'Матрица А:\n{A}')
        # print(f'Матрица B:\n{B}')
        for i in range(len(A[0])):
            A_i = A.copy()
            for j in range(len(A)):
                A_i[j][i] = B[j]
            x_i.append(int(np.linalg.det(A_i) / opred_A))
        print(x_i)
    else:
        print('Система линейных уравнений несовместна')


# number_8()


# Сделал
def number_9():
    A = np.array([[2, -1, 0, -3], [1, 0, -1, 2], [3, -2, 1, -1], [-1, 3, -1, 1]])
    B = np.array([[-9, 8, -5, 9]])
    C = np.concatenate((A, B.T), axis=1)  # расширенная матрица
    if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(C):
        print('Система линейных уравнений совместна')

        D = sympy.Matrix(C).rref() # ступенчатая матрица для Гаусса

        # Первый способ
        S = np.asarray(D[0])
        print(S)
        print(*[S[i][-1] for i in range(len(S))])

        # Второй способ
        G = np.array(D[0].T)
        print(G)
        print(G[-1])
    else:
        print('Система линейных уравнений несовместна')


# number_9()


# Вопрос
def number_10():
    t = sympy.Symbol('t')
    k = sympy.Symbol('k')
    x = sympy.Function('x')
    p = sympy.Function('p')
    # x" = t*x' + x + 1
    # p = x'
    # p' = t*p + x + 1

    # print((sympy.dsolve((sympy.diff(x(t), t) - p(t), -sympy.diff(p(t), t) + t*p(t) + x(t) + 1))))
    print(sympy.dsolve(-sympy.diff(x(t), t, 2) + sympy.diff(x(t), t) * k + x(t) + 1, x(t)))


# number_10()