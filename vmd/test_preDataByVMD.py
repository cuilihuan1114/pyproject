import numpy as np
from vmd.VMD import vmd
import matplotlib.pyplot as plt
from base import preprocess
import _thread
import os
import threading

# VMD的相关参数
alpha = 2000  # moderate bandwidth constraint
tau = 0  # noise-tolerance (no strict fidelity enforcement)
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7


# K：分解的通道数K+1
# length: 加载数据的长度
# load：是否从缓存当中加载
def preprocessByVMD(step=3, K=5, path='newcache1', length=2048, load=True, showResultPlot=False):
    # 判断保存的文件夹是否存在 如果不存在 则创建新的
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    # 保存分解后的文件的路径 ../vmd/cache63
    x_train_vmd_save_path = path + '/x_train_vmd.npy'
    y_train_save_path = path + '/y_train.npy'
    x_valid_vmd_save_path = path + '/x_valid_vmd.npy'
    y_valid_save_path = path + '/y_valid.npy'

    # 原始数据保存路径
    x_train_save_path = path + '/x_train.npy'
    x_valid_save_path = path + '/x_valid.npy'

    # 对原始数据进行原始多尺度数据处理
    x_train_originalscala_save_path = path + '/x_train_originalscala.npy'
    x_valid_originalscala_save_path = path + '/x_valid_originalscala.npy'

    # 通过原始数据处理的多维数据
    x_train_scala_save_path = path + '/x_train_scala.npy'
    x_valid_scala_save_path = path + '/x_valid_scala.npy'

    # 多维数据经过vmd处理后的数据
    x_train_scala_vmd_save_path = path + '/x_train_scala_vmd.npy'
    x_valid_scala_vmd_save_path = path + '/x_valid_scala_vmd.npy'

    # 对原始多维数据进行vmd处理
    x_train_originalscala_vmd_save_path = path + '/x_train_originalscala_vmd.npy'
    x_valid_originalscala_vmd_save_path = path + '/x_valid_originalscala_vmd.npy'

    # 从分解文件当中加载数据
    if load and os.path.exists(x_train_vmd_save_path) and os.path.exists(y_train_save_path) and os.path.exists(
            x_valid_vmd_save_path) and os.path.exists(y_valid_save_path) and os.path.exists(
        x_train_save_path) and os.path.exists(x_valid_save_path) and os.path.exists(x_train_scala_save_path) \
            and os.path.exists(x_train_originalscala_save_path):

        x_train = np.load(x_train_save_path)
        x_valid = np.load(x_valid_save_path)

        y_train = np.load(y_train_save_path)
        y_valid = np.load(y_valid_save_path)

        x_train_vmd = np.load(x_train_vmd_save_path)
        x_valid_vmd = np.load(x_valid_vmd_save_path)

        x_train_originalscala = np.load(x_train_originalscala_save_path)
        x_valid_originalscala = np.load(x_valid_originalscala_save_path)

        x_train_scala = np.load(x_train_scala_save_path)
        x_valid_scala = np.load(x_valid_scala_save_path)

        x_train_originalscala_vmd = np.load(x_train_originalscala_vmd_save_path)
        x_valid_originalscala_vmd = np.load(x_valid_originalscala_vmd_save_path)

        x_train_scala_vmd = np.load(x_train_scala_vmd_save_path)
        x_valid_scala_vmd = np.load(x_valid_scala_vmd_save_path)

        if showResultPlot:
            showPlot(x_train, length, K)
        print("从缓存中加载完毕！")
        return x_train, x_valid, x_train_vmd, x_valid_vmd, x_train_originalscala, x_valid_originalscala, x_train_scala, \
               x_valid_scala, x_train_originalscala_vmd, x_valid_originalscala_vmd, x_train_scala_vmd, x_valid_scala_vmd, y_train, y_valid

    # 加载数据从原始mat当中
    number = 1000  # 每类样本的数量
    normal = True  # 是否标准化
    rate = [0.5, 0.49, 0.01]  # 测试集验证集划分比例
    path = r'E:\实验室资料\1.毕业论文\6.数据集\data\0HP'

    print("===================加载原始数据==========================")
    # 获得原始数据
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path, length=length,
                                                                           number=number,
                                                                           normal=normal,
                                                                           rate=rate,
                                                                           enc=True,
                                                                           enc_step=28
                                                                           )

    print("===================加载完毕！===========================")

    print("===================处理原始多维度数据========================")
    # 获取多维度数据
    x_train_originalscala = processOriginalMultScalas(step, x_train)
    x_valid_originalscala = processOriginalMultScalas(step, x_valid)
    print("===================处理完毕！============================")

    print("===================处理改进后的多维度数据========================")
    # 获取多维度数据
    x_train_scala, x_valid_scala = processMultScals(step, x_train, x_valid)
    print("===================处理完毕！============================")

    x_train_vmd = dataByVMD(x_train, "x_train训练集")
    x_valid_vmd = dataByVMD(x_valid, "x_valid测试集")

    x_train_scala_vmd = dataByVMD(x_train_scala, "x_train_scala训练集")
    x_valid_scala_vmd = dataByVMD(x_valid_scala, "x_valid_scala测试集")

    x_train_originalscala_vmd = dataByVMD(x_train_originalscala, "x_train_originalscala训练集")
    x_valid_originalscala_vmd = dataByVMD(x_valid_originalscala, "x_valid_originalscala测试集")

    print("======================保存文件当中====================")
    # 保存到文件当中 方便下次直接加载
    np.save(x_train_save_path, x_train)
    np.save(x_valid_save_path, x_valid)

    np.save(x_train_vmd_save_path, x_train_vmd)
    np.save(x_valid_vmd_save_path, x_valid_vmd)

    np.save(x_train_scala_save_path, x_train_scala)
    np.save(x_valid_scala_save_path, x_valid_scala)

    np.save(x_train_scala_vmd_save_path, x_train_scala_vmd)
    np.save(x_valid_scala_vmd_save_path, x_valid_scala_vmd)

    np.save(x_train_originalscala_save_path, x_train_originalscala)
    np.save(x_valid_originalscala_save_path, x_valid_originalscala)

    np.save(x_train_originalscala_vmd_save_path, x_train_originalscala_vmd)
    np.save(x_valid_originalscala_vmd_save_path, x_valid_originalscala_vmd)

    np.save(y_train_save_path, y_train)
    np.save(y_valid_save_path, y_valid)
    print("=================缓存保存完毕=============================================")

    if showResultPlot:
        showPlot(x_train, length, K)
    return x_train, x_valid, x_train_vmd, x_valid_vmd, x_train_originalscala, x_valid_originalscala, x_train_scala, \
           x_valid_scala, x_train_originalscala_vmd, x_valid_originalscala_vmd, x_train_scala_vmd, x_valid_scala_vmd, y_train, y_valid


def dataByVMD(train, describe, K=5):
    # 分解训练解
    print("开始分解", describe, "上的数据:,总共:", int(train.shape[0] / 50), "份")
    train_list = []
    for i in range(train.shape[0]):
        # 调用vmd函数
        u, u_hat, omega = vmd(train[i, :], alpha, tau, K, DC, init, tol)  # u是分解后的信号
        if i % 50 == 0:
            print(" >>>正在分解第:", int(i / 50) + 1, "份")
        train_list.append(u.flatten(order='F'))
    return np.array(train_list)


# u表示的是分解后的数据
# row表示的是原始数据
# T表示显示的点数
def showPlot(row, T, K):
    fs = 1 / T
    t = np.linspace(1, T, num=T) / T
    freqs = 2 * np.pi * (t - 0.5 - 1 / T) / fs

    # 调用vmd函数
    u, u_hat, omega = vmd(row[0, :], alpha, tau, K, DC, init, tol)  # u是分解后的信号

    plt.figure(figsize=(1.8 * 8, 2.4 * 4))

    # 配置中文显示
    plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 表示一张图一共的个数 u.shape[0]+1表示分解的个数+1  2表示的是两侧 1表示第一个位置
    plt.subplot(u.shape[0] + 1, 2, 1)  # n_row , n_col
    # 表示的是原始数据
    plt.plot(t, row[0])
    plt.title(u'VMD分解')
    plt.grid()
    # 表示第二个区域
    plt.subplot(u.shape[0] + 1, 2, 2)
    plt.plot(freqs, np.abs(np.fft.fft(row[0])))
    plt.title(u'对应频谱')
    plt.grid()
    for i in range(u.shape[0]):
        # 对应的位置 左边
        plt.subplot(u.shape[0] + 1, 2, i * 2 + 3)
        plt.plot(t, u[i, :])
        plt.grid()
        # 对应的位置 右边
        plt.subplot(u.shape[0] + 1, 2, i * 2 + 4)
        plt.plot(freqs, np.abs(np.fft.fft(u[i, :])))
        plt.grid()
    plt.show()


# 用原始多维度处理
def processOriginalMultScalas(step, x_train):
    # 转化x_train
    x_train_list = []
    for i in range(x_train.shape[0]):
        x_train_list_row = []
        sum = 0
        for j in range(x_train[i].shape[0]):
            if j % step == 0 and j != 0:
                x_train_list_row.append(sum / step)
                sum = 0
            sum = sum + x_train[i][j]
        x_train_list.append(x_train_list_row)
    x_train_scala = np.array(x_train_list)
    return x_train_scala


# 用来多维度处理
def processMultScals(step, x_train, x_valid):
    # 转化x_train
    x_train_list = []
    for i in range(x_train.shape[0]):
        x_train_list_row = []
        for j in range(x_train[i].shape[0] - step):
            sum = 0
            for k in range(j, j + step):
                sum = sum + x_train[i][k]
            x_train_list_row.append(sum / step)
        for j in range(x_train[i].shape[0] - step, x_train[i].shape[0]):
            x_train_list_row.append(x_train[i][j])
        x_train_list.append(x_train_list_row)
    x_train_scala = np.array(x_train_list)

    # 转化x_valid
    x_valid_list = []
    for i in range(x_valid.shape[0]):
        x_valid_list_row = []
        for j in range(x_valid[i].shape[0] - step):
            sum = 0
            for k in range(j, j + step):
                sum = sum + x_valid[i][j]
            x_valid_list_row.append(sum / step)
        for j in range(x_valid[i].shape[0] - step, x_valid[i].shape[0]):
            x_valid_list_row.append(x_valid[i][j])
        x_valid_list.append(x_valid_list_row)
    x_valid_scala = np.array(x_valid_list)
    return x_train_scala, x_valid_scala


class myThread(threading.Thread):
    def __init__(self, threadID, name, step, K, base):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.step = step
        self.K = K
        self.base = base

    def run(self):
        print("开始线程：" + self.name)
        preprocessByVMD(self.step, self.K, self.base)
        print("退出线程：" + self.name)


def startThread():
    base = 'cache'
    K = 7
    thread_list = list()
    while K < 8:
        step = 6
        while step < K:
            base = 'cache/cache'
            base = base + str(K) + str(step)
            threadName = "Thread-" + str(K) + str(step)
            # 启动对应的线程
            try:
                thread_list.append(myThread(step * K, threadName, step, K, base))
            except:
                print("Error: unable to start thread")
            step = step + 1
        K = K + 1

    # 开启每个线程
    for thread in thread_list:
        thread.start()

    # 等待线程执行完毕
    for thread in thread_list:
        thread.join()


if __name__ == "__main__":
    # startThread()
    K = 5
    step = 1
    base = 'cache/cache'
    base = base + str(K) + str(step)
    preprocessByVMD(step=step, K=K, path=base)
