import numpy as np
from random import random, seed, randint
from functools import reduce
seed(42)

print('Номер 1')
def number_1():
    arr = [round(random(), 2) for i in range(int(input('Введите длину массива: ')))]
    print(arr)
    # Решение через map и sum
    arr_new = list(map(lambda x: x * (arr.index(x) + 1), arr))
    print(sum([round(i, 2) for i in arr_new]))
    # Решение через reduce
    sum_arr = reduce(lambda x, y: x + y * (arr.index(y) + 1), arr)
    print(sum_arr)
number_1()


print('\nНомер 2')
# Решение с помощью numpy
A = np.random.randint(1, 11, size=(4, 4))
print(A)

# Решение через циклы
n, m = 4, 4
a = [[randint(1, 11) for j in range(m)] for i in range(n)]
print(a)


print('\nНомер 3')
# axis=1 - это строки, axis=0 - это столбцы
A = np.array([[12, -2, 0, 3], [2, 10, 2, 0], [0.5, 4, -7, 6], [1, 7, 5, -3]])  # можно добавить 'int'
print(f'Матрица А:\n{A}')
B = np.mean(A, 1)  # можно добавить 'int'
print(f'Массив B: {B}')
C = np.max(A, 0)
print(f'Массив C: {C}')
D = B * C
print(f'Массив D: {D}')


print('\nНомер 4')
E = A
E[0] = C
E[:, 2] = D
print(f'Матрица E:\n{E}')


print('\nНомер 5')
F = 2 * np.square(A + E) - A * np.linalg.matrix_power(E, -1)
print(f'Матрица F:\n{F}')
print(f'Определитель матрицы F = {round(np.linalg.det(F), 2)}')
print(f'След матрицы F = {round(np.trace(F), 2)}')
print(f'Ранг матрицы F = {np.linalg.matrix_rank(F)}')
print(f'Евклидова норма матрицы F = {round(np.linalg.norm(F), 2)}')


print('\nНомер 6')
X = np.diagonal(F)  # либо F.diagonal()
print(f'Главная диагональ матрицы F: {X}')


print('\nНомер 7')
Y = np.random.normal(0, 1, (16,1))
print(f'Матрица Y, с нормальным распределением рандомных чисел:\n{Y}')
Y_sorted = np.sort(Y, axis=0)
print(f'Матрица Y, отсортированная по убыванию:\n{Y_sorted[::-1]}')


print('\nНомер 8')
Z = A.reshape(16, 1)
print(f'Массив Z\n: {Z}')
res = reduce(lambda x, y: x * y, (Y + Z)) - F[1][2]
print(f'Результат: {res}')


print('\nНомер 9')
P = np.roots([2, 3, 4, 12, 5, 6, 5])
print(f'Корни уравнения: {P}')


print('\nНомер 10')
P1 = np.poly([2, 4, 8, 16, 32])
print(f'Коэффициенты полинома: {P1}')
