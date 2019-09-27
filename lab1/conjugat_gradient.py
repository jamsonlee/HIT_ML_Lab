import numpy as np

from lab1.Draw import draw_fitting_curve
from lab1.GenerateDataWithNoise import generate
import matplotlib.pyplot as plt


print('please input data size and standard deviation of noise !')
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
my_lambda = 0


def diff(theta):
    """
    计算有正则项的偏导
    :param theta: 要拟合的多项式参数
    :return: 现在拟合的多项式的偏导
    """
    return (data_x.T.dot(data_x).dot(theta) - data_x.T.dot(data_y) + 1 / 2 * my_lambda * (
            (theta.T.dot(theta)) ** (-1 / 2))) / data_x.shape[1]


theta = np.zeros((poly_times, 1))
theta[:, 0] = 10 ** -4  # 初始化
Q = (data_x.T.dot(data_x) + my_lambda * np.mat(np.eye(data_x.shape[1]))) / data_x.shape[1]
r = -diff(theta)
p = r
for i in range(1, data_x.shape[1]):
    a = float((r.T.dot(r)) / (p.T.dot(Q).dot(p)))  # learning rate
    r_prev = r
    theta = theta + a * p  # update theta
    r = r - a * Q.dot(p)  # 每次迭代的误差
    p = r + float((r.T.dot(r)) / (r_prev.T.dot(r_prev))) * p  # 共轭向量


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
print(test_error())
print('end')
