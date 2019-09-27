import numpy


def generate(x_size, sigma, interval):  # 生成带有大小为data_size，均值为0，方差为sigma的正弦函数数据集
    """
    生成具有高斯噪声数据的正弦函数的数据
    :param x_size 生成数据的x数量
    :param sigma 所添加的高斯噪声的正态分布方差
    :param interval: 每个数据之间的间隔
    :return y 返回一个包含与x相应函数值y的列表
    """
    x = numpy.arange(1, int(x_size + 1), interval)  # 为了测试test_error改变了步长
    random = numpy.random.normal(0, sigma, int(x_size * (1 / interval)))
    y = numpy.sin(x) + random
    return y
