import numpy as np
from sympy import *
import matplotlib.pyplot as plt


def func(x1, x2):
    return 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2


x0 = [-1.2, 0]
eps = 0.0001
rounding_accuracy = 5


def my_round(arr, accuracy=rounding_accuracy):
    for index, value in np.ndenumerate(arr):
        arr[index] = round(value, accuracy)
    return arr


def central_derivatives_1(func, x0, h, f_x):
    x1 = x0[0]
    x2 = x0[1]

    f1 = func(x1 + h, x2)
    f2 = func(x1 - h, x2)
    f3 = func(x1, x2 + h)
    f4 = func(x1, x2 - h)

    df_dx1 = (f1 - f2) / (2 * h)
    df_dx2 = (f3 - f4) / (2 * h)
    return [[df_dx1, df_dx2], [f_x, f1, f2, f3, f4, df_dx2]]


def central_derivatives_2(func, x0, h, param):
    x1 = x0[0]
    x2 = x0[1]
    f_x, f1, f2, f3, f4, df_dx2 = param

    df2_dx1 = (f1 - 2 * f_x + f2) / (h ** 2)
    df2_dx2 = (f3 - 2 * f_x + f4) / (h ** 2)

    xm1 = x1 + h
    fm3 = func(xm1, x2 + h)
    fm4 = func(xm1, x2 - h)
    dfm_dx2 = (fm3 - fm4) / (2 * h)
    d2f_dxdy = (dfm_dx2 - df_dx2) / h
    res = np.array([df2_dx1, df2_dx2, d2f_dxdy])
    return res


def right_derivatives_1(func, x0, h, f_x):
    x1 = x0[0]
    x2 = x0[1]

    f1 = func(x1 + h, x2)
    f2 = func(x1, x2 + h)

    df_dx1 = (f1 - f_x) / h
    df_dx2 = (f2 - f_x) / h

    return [[df_dx1, df_dx2], [f_x, f1, f2, df_dx2]]


def right_derivatives_2(func, x0, h, param):
    x1 = x0[0]
    x2 = x0[1]
    f3 = func(x1 + 2 * h, x2)
    f4 = func(x1, x2 + 2 * h)

    f_x, f1, f2, df_dx2 = param

    df2_dx1 = (f3 - 2 * f1 + f_x) / (h ** 2)
    df2_dx2 = (f4 - 2 * f2 + f_x) / (h ** 2)

    xm1 = x1 + h
    fm1 = func(xm1, x2)
    fm2 = func(xm1, x2 + h)

    dfm_dx2 = (fm2 - fm1) / h
    d2f_dxdy = (dfm_dx2 - df_dx2) / h
    res = np.array([df2_dx1, df2_dx2, d2f_dxdy])
    return res


def left_derivatives_1(func, x0, h, f_x):
    x1 = x0[0]
    x2 = x0[1]

    f1 = func(x1 - h, x2)
    f2 = func(x1, x2 - h)

    df_dx1 = (f_x - f1) / h
    df_dx2 = (f_x - f2) / h

    return [[df_dx1, df_dx2], [f_x, f1, f2, df_dx2]]


def left_derivatives_2(func, x0, h, param):
    x1 = x0[0]
    x2 = x0[1]
    f3 = func(x1 - 2 * h, x2)
    f4 = func(x1, x2 - 2 * h)

    f_x, f1, f2, df_dx2 = param

    df2_dx1 = (f_x - 2 * f1 + f3) / (h ** 2)
    df2_dx2 = (f_x - 2 * f2 + f4) / (h ** 2)

    xm1 = x1 - h
    fm1 = func(xm1, x2)
    fm2 = func(xm1, x2 - h)

    dfm_dx2 = (fm1 - fm2) / h
    d2f_dxdy = (df_dx2 - dfm_dx2) / h
    res = np.array([df2_dx1, df2_dx2, d2f_dxdy])
    return res


def Newtons_method1(func, x0, eps, h, derivatives_1, derivatives_2, accuracy=rounding_accuracy):
    h = 0.00002
    x_pr = x0
    x_list = [x_pr]
    f_x_list = []
    number_of_iterations = 0
    func_ev_number = 0
    if derivatives_1.__name__ == 'central_derivatives_1':
        div1_ev_num = 4
        div2_ev_num = 2
    else:
        div1_ev_num = 2
        div2_ev_num = 4
    while True:
        x1 = x_pr[0]
        x2 = x_pr[1]
        f_x_pr = func(x1, x2)
        f_x_list.append(f_x_pr)
        func_ev_number += 1
        derivatives_res_1 = derivatives_1(func, x_pr, h, f_x_pr)
        df_dx1, df_dx2 = derivatives_res_1[0]
        func_ev_number += div1_ev_num
        param = derivatives_res_1[1]
        gradient_f = np.array([df_dx1, df_dx2])
        gr_norm = np.linalg.norm(gradient_f)
        if gr_norm <= eps:
            break
        number_of_iterations += 1
        df2_dx1, df2_dx2, d2f_dx1dx2 = derivatives_2(func, x_pr, h, param)
        func_ev_number += div2_ev_num
        H = [[df2_dx1, d2f_dx1dx2], [d2f_dx1dx2, df2_dx2]]
        H1 = np.linalg.inv(H)
        x_next = x_pr - H1 @ gradient_f
        x_list.append(x_next)
        x_pr = x_next

    return [x_list, f_x_list, func_ev_number]


def Newtons_method2(func, x0, eps, h, derivatives_1, derivatives_2):
    x_pr = x0
    x_list = [x_pr]
    number_of_iterations = 1
    f_x_pr = func(x_pr[0], x_pr[1])
    f_x_list = [f_x_pr]
    while True:
        x1 = x_pr[0]
        x2 = x_pr[1]
        derivatives_res_1 = derivatives_1(func, x_pr, h, f_x_pr)
        df_dx1, df_dx2 = derivatives_res_1[0]
        param = derivatives_res_1[1]
        gradient_f = np.array([df_dx1, df_dx2])

        df2_dx1, df2_dx2, d2f_dx1dx2 = derivatives_2(func, x_pr, h, param)
        H = [[df2_dx1, d2f_dx1dx2], [d2f_dx1dx2, df2_dx2]]
        H1 = np.linalg.inv(H)
        x_next = x_pr - H1 @ gradient_f

        x_list.append(x_next)
        f_x_next = func(x_next[0], x_next[1])

        if np.linalg.norm(np.array(x_next - x_pr)) / np.linalg.norm(np.array(x_pr)) <= eps and abs(
                f_x_next - f_x_pr) <= eps:
            break
        x_pr = x_next
        f_x_pr = f_x_next
        f_x_list.append(f_x_pr)
        number_of_iterations += 1
    return [x_list, f_x_list, number_of_iterations * 7]


h = 0.0001
x11, y11, function_evaluations_number11 = Newtons_method1(func, x0, eps, h, central_derivatives_1,
                                                          central_derivatives_2)
x12, y12, function_evaluations_number12 = Newtons_method1(func, x0, eps, h, right_derivatives_1, right_derivatives_2)
x13, y13, function_evaluations_number13 = Newtons_method1(func, x0, eps, h, left_derivatives_1, left_derivatives_2)

h = 0.00001
x21, y21, function_evaluations_number21 = Newtons_method1(func, x0, eps, h, central_derivatives_1,
                                                          central_derivatives_2)
x22, y22, function_evaluations_number22 = Newtons_method1(func, x0, eps, h, right_derivatives_1, right_derivatives_2)
x23, y23, function_evaluations_number23 = Newtons_method1(func, x0, eps, h, left_derivatives_1, left_derivatives_2)

h = 0.000015
x31, y31, function_evaluations_number31 = Newtons_method1(func, x0, eps, h, central_derivatives_1,
                                                          central_derivatives_2)
x32, y32, function_evaluations_number32 = Newtons_method1(func, x0, eps, h, right_derivatives_1, right_derivatives_2)
x33, y33, function_evaluations_number33 = Newtons_method1(func, x0, eps, h, left_derivatives_1, left_derivatives_2)

h = 0.0001
x211, y211, function_evaluations_number211 = Newtons_method2(func, x0, eps, h, central_derivatives_1,
                                                             central_derivatives_2)
x212, y212, function_evaluations_number212 = Newtons_method2(func, x0, eps, h, right_derivatives_1, right_derivatives_2)
x213, y213, function_evaluations_number213 = Newtons_method2(func, x0, eps, h, left_derivatives_1, left_derivatives_2)

h = 0.00001
x221, y221, function_evaluations_number221 = Newtons_method2(func, x0, eps, h, central_derivatives_1,
                                                             central_derivatives_2)
x222, y222, function_evaluations_number222 = Newtons_method2(func, x0, eps, h, right_derivatives_1, right_derivatives_2)
x223, y223, function_evaluations_number223 = Newtons_method2(func, x0, eps, h, left_derivatives_1, left_derivatives_2)

h = 0.000015
x231, y231, function_evaluations_number231 = Newtons_method2(func, x0, eps, h, central_derivatives_1,
                                                             central_derivatives_2)
x232, y232, function_evaluations_number232 = Newtons_method2(func, x0, eps, h, right_derivatives_1, right_derivatives_2)
x233, y233, function_evaluations_number233 = Newtons_method2(func, x0, eps, h, left_derivatives_1, left_derivatives_2)
#
# print('''\n*************************************************
# *************************************************
# *************************************************\n''')
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number11,
                                                                                        x11[-1], y11[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number12,
                                                                                        x12[-1], y12[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number13,
                                                                                        x13[-1], y13[-1]))
print('--------------------------------------------------------')

print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number21,
                                                                                        x21[-1], y21[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number22,
                                                                                        x22[-1], y22[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number23,
                                                                                        x23[-1], y23[-1]))
print('--------------------------------------------------------')

print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number31,
                                                                                        x31[-1], y31[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number32,
                                                                                        x32[-1], y32[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number33,
                                                                                        x33[-1], y33[-1]))
print('------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------')

print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number211,
                                                                                        x211[-1], y211[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number212,
                                                                                        x212[-1], y212[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number213,
                                                                                        x213[-1], y213[-1]))
print('--------------------------------------------------------')

print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number221,
                                                                                        x221[-1], y221[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number222,
                                                                                        x222[-1], y222[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number223,
                                                                                        x223[-1], y223[-1]))
print('--------------------------------------------------------')

print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number231,
                                                                                        x231[-1], y231[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number232,
                                                                                        x232[-1], y232[-1]))
print('Кількість обчислень значень функції: {}.  Точка мінімуму: {}   f(x) = {}'.format(function_evaluations_number233,
                                                                                        x233[-1], y233[-1]))
print('------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------')
