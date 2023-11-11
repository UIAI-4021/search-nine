import pandas as pd
import numpy as np
import random

w1, w2, w3, w4, w5, w6 = -1, -0.5, 0, 0.5, 1, 0.2


def f(x1, x2, x3, x4, x5, x6):
    return w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6


def df_dx1():
    return w1


def df_dx2():
    return w2


def df_dx3():
    return w3


def df_dx4():
    return w4


def df_dx5():
    return w5


def df_dx6():
    return w6


def gradient_descent(start_x1, start_x2, start_x3, start_x4, start_x5, start_x6
                     , learning_rate, num_iterations):
    history = []

    x1 = start_x1
    x2 = start_x2
    x3 = start_x3
    x4 = start_x4
    x5 = start_x5
    x6 = start_x6

    for i in range(num_iterations):
        x1 = x1 - learning_rate * df_dx1()
        x2 = x2 - learning_rate * df_dx2()
        x3 = x3 - learning_rate * df_dx3()
        x4 = x4 - learning_rate * df_dx4()
        x5 = x5 - learning_rate * df_dx5()
        x6 = x6 - learning_rate * df_dx6()

        history.append([x1, x2, x3, x4, x5, x6, f(x1, x2, x3, x4, x5, x6)])

    return x1, x2, x3, x4, x5, x6, f(x1, x2, x3, x4, x5, x6), history


if __name__ == '__main__':
    dataset = pd.read_csv('Flight_Price_Dataset_Q2.csv')




