import numpy as np
import matplotlib.pyplot as plt


def function_y(x, poly_times, theta):
    sum = 0
    for i in np.arange(0, poly_times):
        sum += (x ** i) * theta[i, 0]
    return sum


def draw_fitting_curve(data_size, x_size, poly_times, theta):
    x1 = np.arange(1, data_size + 1, 0.01)
    y = function_y(x1, poly_times, theta)
    plt.plot(x1, y)
    plt.xlim(0, x_size + 1)
    plt.ylim(-2, 2)
    plt.show()


def draw_cost_history(cost_value, times):
    x = np.arange(1, cost_value.size + 1)
    plt.plot(x, cost_value)
    plt.xlim(0, times)
    plt.show()