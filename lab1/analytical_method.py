import numpy as np
from lab1.GenerateDataWithNoise import generate
import matplotlib.pyplot as plt

print('please input data size and standard deviation of noise')
data_size = int(input())
sigma = float(input())
print('请输入每个数据之间的间隔')
interval = float(input())
x_size = data_size*interval
list_y = np.transpose(generate(x_size, sigma, interval))
data_x = np.zeros((data_size, data_size))
data_x[:, 0] = 1
data_x[:, 1] = np.arange(1, int(x_size+1), interval)
data_y = list_y.reshape(data_size, 1)
plt.show()  # 绘制散点图
for j in range(2, data_size):
    data_x[:, j] = data_x[:, j - 1] * data_x[:, 1]  # data_x的求高阶项
plt.scatter(data_x[:, 1], data_y)
print('是否使用正则项？（1.不使用   2，使用）')
if input() == '1':
    theta = np.linalg.inv(np.transpose(data_x).dot(data_x)).dot(np.transpose(data_x)).dot(data_y)
else:
    print('输入lambda的大小')
    my_lambda = float(input())
    theta = np.linalg.inv(np.transpose(data_x).dot(data_x) + my_lambda * np.eye(data_size)).dot(
        np.transpose(data_x)).dot(data_y)

print(theta)


def function_y(x, theta):
    """
    计算拟合多项式x对应的y值
    :param x: 自变量的值
    :param theta: 多项式各个参数的值(向量)
    :return: 与输入多项式的参数和x相对应的y值
    """
    sum = 0
    for i in np.arange(0, data_size):
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
    return sum_error/(x_size*10)


x = np.arange(1, data_size + 1, 0.01)
y = function_y(x, theta)
plt.plot(x, y)
plt.xlim(0, data_size + 1)
plt.ylim(-2, 2)
plt.show()  # 绘制拟合的多项式的图像
test_errors = 0
for i in range(100):
    test_errors= test_errors+test_error()
print(test_error()/100)
