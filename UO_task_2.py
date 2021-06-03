import numpy as np
from sympy import *
import matplotlib.pyplot as plt

eps = 0.0001
rounding_accuracy = 5

R = 1000


def func11(x1, x2):
    global R
    res = 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2 - R * ln(4 - x1 ** 2 - x2 ** 2)  # 1
    return res


def condition11(x):
    x1 = x[0]
    x2 = x[1]
    radius = 2
    centre = [0, 0]
    if np.sqrt((x1 - centre[0]) ** 2 + (x2 - centre[1]) ** 2) < radius:
        return True
    else:
        return False


def func12(x1, x2):
    global R
    res = 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2 - R * ln(16 - x1 ** 2 - (x2 + 4) ** 2)
    return res


def condition12(x):
    x1 = x[0]
    x2 = x[1]
    radius = 4
    centre = [0, -4]
    if np.sqrt((x1 - centre[0]) ** 2 + (x2 - centre[1]) ** 2) < radius:
        return True
    else:
        return False



def func21(x1, x2):
    global R
    res = 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2 - R * ln(49 - (x1 ** 2) - ((x2 + 4) ** 2)) - R * ln((x1 ** 2) + ((x2 + 4) ** 2) - 1)

    return res


def condition21(x):
    x1 = x[0]
    x2 = x[1]
    if (x1 ** 2) + ((x2 + 4) ** 2) < 49 and (x1 ** 2) + ((x2 + 4) ** 2) > 1:
        return True
    else:
        return False


def my_round(arr):
    for index, value in np.ndenumerate(arr):
        arr[index] = round(value, 5)
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


def internal_point_method(func, x0, eps, derivatives_1, derivatives_2, area_conditions):
    h = 0.000015

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

        try:
            H = np.array(H, dtype=float)
        except:
            global flag
            flag = 1

            return [x_list, f_x_list, number_of_iterations * 7]
        H1 = np.linalg.inv(H)

        n = 1
        number_of_attempts = 0
        while True:
            x_next = my_round(x_pr - H1 @ gradient_f / n)
            for i in range(2):
                x_next[i] = float(x_next[i])

            x_list.append(x_next)
            if area_conditions(x_next):
                break
            n *= 2
            number_of_attempts += 1

        f_x_next = func(x_next[0], x_next[1])
        if np.linalg.norm(np.array(x_next - x_pr)) / np.linalg.norm(np.array(x_pr)) <= eps and abs(
                f_x_next - f_x_pr) <= eps:
            global R
            R = R / 10
            break

        x_pr = x_next
        f_x_pr = f_x_next
        f_x_list.append(f_x_pr)
        number_of_iterations += 1

    return [x_list, f_x_list, number_of_iterations * 7]


# ------------------------------------------------------------------

point_x1 = [-1.2, 0]
res_list1 = [point_x1]
res_y_list1 = [func11(point_x1[0], point_x1[1])]
flag = 0
total_f_ev_number1 = 0
while True:
    x, y, function_evaluations_number = internal_point_method(func11, point_x1, eps, central_derivatives_1, central_derivatives_2, condition11)
    res_list1.append(x[-1])
    res_y_list1.append(y[-1])
    total_f_ev_number1 += function_evaluations_number
    if (np.linalg.norm(np.array(res_list1[-1]-res_list1[-2]))/np.linalg.norm(np.array(res_list1[-2])) <= eps and abs(res_y_list1[-1]-res_y_list1[-2])<= eps) or flag == 1 :
        break


R = 1000
point_x2 = [1, -2]
res_list2 = [point_x2]
res_y_list2 = [func12(point_x2[0], point_x2[1])]
flag = 0
total_f_ev_number2 = 0
while True:
    x, y, function_evaluations_number = internal_point_method(func12, point_x2, eps, central_derivatives_1, central_derivatives_2, condition12)
    res_list2.append(x[-1])
    res_y_list2.append(y[-1])
    total_f_ev_number2 += function_evaluations_number
    if (np.linalg.norm(np.array(res_list2[-1]-res_list2[-2]))/np.linalg.norm(np.array(res_list2[-2])) <= eps and abs(res_y_list2[-1]-res_y_list2[-2])<= eps) or flag == 1 :
        break

R = 1000
point_x3 = [-1, -1]
res_list3 = [point_x3]
res_y_list3 = [func12(point_x3[0], point_x3[1])]
flag = 0
total_f_ev_number3 = 0
while True:
    x, y, function_evaluations_number = internal_point_method(func21, point_x3, eps, central_derivatives_1, central_derivatives_2, condition21)
    res_list3.append(x[-1])
    res_y_list3.append(y[-1])
    total_f_ev_number3 += function_evaluations_number
    if (np.linalg.norm(np.array(res_list3[-1] - res_list3[-2])) / np.linalg.norm(np.array(res_list3[-2])) <= eps and abs(res_y_list3[-1] - res_y_list3[-2]) <= eps) or flag == 1:
        break



print('Точка мінімуму: ', my_round(res_list1[-1]))
print('Кількість обчислень значень функції: ', total_f_ev_number1)

print('-------------------')
print('Точка мінімуму: ', my_round(res_list2[-1]))
print('Кількість обчислень значень функції: ', total_f_ev_number2)

print('-------------------')
print('Точка мінімуму: ', my_round(res_list3[-1]))
print('Кількість обчислень значень функції: ', total_f_ev_number3)

fig, ax = plt.subplots()
res_list1 = np.array(res_list1).T
ax.plot(res_list1[0], res_list1[1], marker='v', markersize=4)
circle1 = plt.Circle((0, 0), 2, color='g', fill=False)
ax = plt.gca()
ax.add_patch(circle1)
plt.show()

fig, ax = plt.subplots()
res_list2 = np.array(res_list2).T
ax.plot(res_list2[0], res_list2[1], marker='v', markersize=4)
circle1 = plt.Circle((0, -4), 4, color='g', fill=False)
ax = plt.gca()
ax.add_patch(circle1)
plt.show()



fig, ax = plt.subplots()
res_list1 = np.array(res_list3).T
ax.plot(res_list1[0], res_list1[1], marker='v', markersize=4)
circle1 = plt.Circle((0, -4), 7, color='g', fill=False)
circle2 = plt.Circle((0, -4), 1, color='g', fill=False)
ax = plt.gca()
ax.add_patch(circle1)
ax.add_patch(circle2)
plt.show()
