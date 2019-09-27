import numpy as np

from lab1.Draw import draw_fitting_curve, draw_cost_history
from lab1.GenerateDataWithNoise import generate
import matplotlib.pyplot as plt

print('please input data size and standard deviation of noise')
data_size = int(input())  # 训练数据的数量
sigma = float(input())  # 高斯噪声的正态分布方差
print('请输入每个数据之间的间隔')
interval = float(input())  # 训练数据之间的间隔
x_size = int(data_size*interval)
print("请输入要拟合的多项式次数")
poly_times = int(input())  # 要拟合的多项式次数
list_y = np.transpose(generate(x_size, sigma, interval))
data_x = np.zeros((data_size, poly_times))
data_x[:, 0] = 1
data_x[:, 1] = np.arange(1, int(data_size*interval+1), interval)
data_y = list_y.reshape(data_size, 1)  # 训练数据的函数值
plt.show()
for j in range(2, poly_times):
    data_x[:, j] = data_x[:, j - 1] * data_x[:, 1]  # 求高阶项
plt.scatter(data_x[:, 1], data_y)  # 绘制训练数据的散点图
my_lambda = 0.001


def cost(theta):
    """
    计算无正则项的误差
    :param theta: 要拟合的多项式参数
    :return: 现在拟合的多项式的误差
    """
    return np.transpose(data_x.dot(theta) - data_y).dot(data_x.dot(theta) - data_y) / (2*data_size)


def cost_with_regularization(theta):
    """
    计算有正则项的误差
    :param theta: 要拟合的多项式参数
    :return: 现在拟合的多项式的误差
    """
    return ((data_x.dot(theta)-data_y).T.dot(data_x.dot(theta) - data_y)+ my_lambda*theta.T.dot(theta))/(2*data_size)


def diff(theta):
    """
    计算无正则项的偏导
    :param theta: 要拟合的多项式参数
    :return: 现在拟合的多项式的偏导
    """
    return (data_x.dot(theta) - data_y).T.dot((data_x.reshape(data_size, poly_times))) / data_size


def diff_with_regularization(theta):
    """
    计算有正则项的偏导
    :param theta: 要拟合的多项式参数
    :return: 现在拟合的多项式的偏导
    """
    return ((data_x.dot(theta) - data_y).T.dot((data_x.reshape(data_size, poly_times))) + 1 / 2 * my_lambda * ((theta.T.dot(theta)) ** (-1 / 2))) / data_size


theta = np.zeros((poly_times, 1))
theta[:, 0] = 10**(-4)
old_cost = cost_with_regularization(theta)
times = 0
max_time = (10**(7))
cost_value = np.zeros((max_time + 1, 1))
cost_value[times, 0] = old_cost
rate = 3*(10 ** (-7))
theta_history = [theta.reshape(poly_times, 1)]
while True:
    dir = np.zeros((1, poly_times))
    dir[0, :] = diff_with_regularization(theta)
    theta = theta - rate * np.transpose(dir)
    new_cost = cost_with_regularization(theta)
    cost_value[times, 0] = new_cost[0]
    times = times + 1
    theta_history.append(theta.reshape(poly_times, 1))
    if abs(old_cost - new_cost) < 10 ** (-10) or times > max_time or new_cost> 10**40:
        break
    old_cost = new_cost


def function_y(x, theta):
    """
    计算拟合多项式x对应的y值
    :param x: 自变量的值
    :param theta: 多项式各个参数的值(向量)
    :return: 与输入多项式的参数和x相对应的y值
    """
    sum = 0
    for i in np.arange(0, poly_times):
        sum += (x ** i) * theta[i, 0]
    return sum


def test_error():
    """
    计算拟合多项式的测试集误差
    :return: 单个x的平均误差
    """
    x = np.arange(1, x_size + 1, 0.1)
    test_y = generate(x_size, sigma, 0.1)
    sum_error = np.transpose(test_y - function_y(x, theta)).dot(test_y - function_y(x, theta))
    return sum_error / (x_size * 10)


draw_fitting_curve(data_size, x_size, poly_times, theta)
draw_cost_history(cost_value, times)
print(test_error())
