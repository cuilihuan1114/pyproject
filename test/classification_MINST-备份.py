import tensorflow.compat.v1 as tf
import vmd.preDataByVMD as preDataByVMD

from models.double.doubledbn import DoubleDBN

tf.compat.v1.disable_eager_execution()

import numpy as np
from base import preprocess

np.random.seed(1337)  # for reproducibility

import os
import gc
import datetime
import time
# 每隔5次打印一下系统的变量使用情况
import objgraph

# 取消打印红色日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
filename = os.path.basename(__file__)

from models.dbn import DBN
from base.base_func import run_sess, run_sess1

from pso.PSO import PSO

# =======================加载完毕==============================
tf.reset_default_graph()


def splitDatas(data, num, all):
    cols = data.shape[1]
    length = cols / all
    end = length * num
    arr = np.hsplit(data, (int(end), cols))
    return arr[0], arr[1]


# 打印当前内存信息
count = 0


def printMemoryInfo():
    print()
    print()
    print()
    print("=====================当前的次数:=========================")
    global count
    count = count + 1
    print(count)
    print("========================================================")
    if count % 5 == 0:
        print("=====================展示当前的内存以及垃圾回收===========")
        objgraph.show_growth()  # show the growth of the objects
        gc.collect()
        print("======================================================")


# 使用DBN训练
def singleDBN(datasets, struct):
    classifier = DBN(
        # hidden_act_func='sigmoid',
        hidden_act_func='relu',
        output_act_func='softmax',
        loss_func='cross_entropy',  # gauss 激活函数会自动转换为 mse 损失函数
        struct=struct,
        lr=1e-3,
        momentum=0.5,
        use_for='classification',  # 表示用来分类
        bp_algorithm='rmsp',  # 优化器选择
        epochs=20,  # 迭代次数
        batch_size=16,
        dropout=0.12,
        units_type=['gauss', 'bin'],
        rbm_lr=1e-3,
        rbm_epochs=10,
        cd_k=1,
        pre_train=True)
    ros = run_sess(classifier, datasets, filename, load_saver='')
    return ros


def doubleDBN(datasets, datasets1, struct, struct1):
    classifier = DoubleDBN(
        hidden_act_func='relu',
        output_act_func='softmax',
        loss_func='cross_entropy',  # gauss 激活函数会自动转换为 mse 损失函数
        struct=struct,
        struct1=struct1,
        lr=1e-3,
        momentum=0.5,
        use_for='classification',  # 表示用来分类
        # bp_algorithm='rmsp',
        bp_algorithm='rmsp',
        epochs=20,  # 迭代次数 20
        batch_size=16,
        dropout=0.12,
        units_type=['gauss', 'bin'],
        rbm_lr=1e-3,
        rbm_epochs=10,  # 10
        cd_k=1,
        pre_train=True)
    ros = run_sess1(classifier, datasets, datasets1, filename, load_saver='')
    return ros


def train_single_dbn(struct,):
    print("=============开始训练单DBN原始数据=================")

    x1, x2, x3 = struct
    train_datasets = [x_train, y_train, x_valid, y_valid]
    train_x_dim = train_datasets[0].shape[1]
    train_y_dim = train_datasets[1].shape[1]
    acc = singleDBN(train_datasets, [train_x_dim, x1, x2, x3, train_y_dim])
    print("====================单DBN原始数据训练完毕！！！===================")

    return acc


def train_double_dbn(struct, struct1, num, all):
    print("=============开始训练双DBN原始数据=================")

    x1_1, x1_2, x1_3 = struct
    x2_1, x2_2, x2_3 = struct1
    x_train_split_1, x_train_split_2 = splitDatas(x_train, num, all)
    x_valid_split_1, x_valid_split_2 = splitDatas(x_valid, num, all)

    train_datasets = [x_train_split_1, y_train, x_valid_split_1, y_valid]
    train_datasets1 = [x_train_split_2, y_train, x_valid_split_2, y_valid]

    train_x_dim = train_datasets[0].shape[1]
    train_y_dim = train_datasets[1].shape[1]
    train_x_dim_1 = train_datasets1[0].shape[1]
    train_y_dim_1 = train_datasets1[1].shape[1]

    struct = [train_x_dim, x1_1, x1_2, x1_3, train_y_dim]
    struct1 = [train_x_dim_1, x2_1, x2_2, x2_3, train_y_dim_1]

    acc = doubleDBN(train_datasets, train_datasets1, struct, struct1)
    print("====================双DBN原始数据训练完毕！！！===================")

    return acc


def train_vmd_single_dbn(struct):
    print("=================开始训练单DBNVMD分解=======================")

    x1, x2, x3 = struct
    train_vmd_datasets = [x_train_vmd, y_train, x_valid_vmd, y_valid]
    train_vmd_x_dim = train_vmd_datasets[0].shape[1]
    train_vmd_y_dim = train_vmd_datasets[1].shape[1]
    acc = singleDBN(train_vmd_datasets, [train_vmd_x_dim, x1, x2, x3, train_vmd_y_dim])
    print("=================单DBNVMD分解训练完毕！！！======================")

    return acc


def train_vmd_double_dbn(struct, struct1, num, all):
    print("=================开始训练双DBNVMD分解=======================")

    x1_1, x1_2, x1_3 = struct
    x2_1, x2_2, x2_3 = struct1
    x_train_vmd_split_1, x_train_vmd_split_2 = splitDatas(x_train_vmd, num, all)
    x_valid_vmd_split_1, x_valid_vmd_split_2 = splitDatas(x_valid_vmd, num, all)

    train_datasets = [x_train_vmd_split_1, y_train, x_valid_vmd_split_1, y_valid]
    train_datasets1 = [x_train_vmd_split_2, y_train, x_valid_vmd_split_2, y_valid]

    train_x_dim = train_datasets[0].shape[1]
    train_y_dim = train_datasets[1].shape[1]
    train_x_dim_1 = train_datasets1[0].shape[1]
    train_y_dim_1 = train_datasets1[1].shape[1]

    struct = [train_x_dim, x1_1, x1_2, x1_3, train_y_dim]
    struct1 = [train_x_dim_1, x2_1, x2_2, x2_3, train_y_dim_1]

    acc = doubleDBN(train_datasets, train_datasets1, struct, struct1)
    print("=================双DBNVMD分解训练完毕！！！======================")

    return acc


def train_originalscala_single_dbn(struct):
    print("=================开始使用单DBN原始多尺度分解======================")

    x1, x2, x3 = struct
    train_datasets = [x_train_originalscala, y_train, x_valid_originalscala, y_valid]
    train_originalscala_x_dim = train_datasets[0].shape[1]
    train_originalscala_y_dim = train_datasets[1].shape[1]
    acc = singleDBN(train_datasets, [train_originalscala_x_dim, x1, x2, x3, train_originalscala_y_dim])
    print("=================使用单DBN多尺度训练完毕======================")
    return acc


def train_originalscala_double_dbn(struct, struct1, num, all):
    print("=================开始使用双DBN原始多尺度======================")

    x1_1, x1_2, x1_3 = struct
    x2_1, x2_2, x2_3 = struct1
    x_train_originalscala_split_1, x_train_originalscala_split_2 = splitDatas(x_train_originalscala, num, all)
    x_valid_originalscala_split_1, x_valid_originalscala_split_2 = splitDatas(x_valid_originalscala, num, all)

    train_datasets = [x_train_originalscala_split_1, y_train, x_valid_originalscala_split_1, y_valid]
    train_datasets1 = [x_train_originalscala_split_2, y_train, x_valid_originalscala_split_2, y_valid]

    train_x_dim = train_datasets[0].shape[1]
    train_y_dim = train_datasets[1].shape[1]
    train_x_dim_1 = train_datasets1[0].shape[1]
    train_y_dim_1 = train_datasets1[1].shape[1]

    struct = [train_x_dim, x1_1, x1_2, x1_3, train_y_dim]
    struct1 = [train_x_dim_1, x2_1, x2_2, x2_3, train_y_dim_1]

    acc = doubleDBN(train_datasets, train_datasets1, struct, struct1)
    print("=================使用双DBN多尺度训练完毕======================")
    return acc


def train_scala_single_dbn(struct):
    print("=================开始使用单DBN多尺度======================")

    x1, x2, x3 = struct
    train_scala_datasets = [x_train_scala, y_train, x_valid_scala, y_valid]
    train_scala_x_dim = train_scala_datasets[0].shape[1]
    train_scala_y_dim = train_scala_datasets[1].shape[1]
    acc = singleDBN(train_scala_datasets, [train_scala_x_dim, x1, x2, x3, train_scala_y_dim])
    print("=================使用单DBN多尺度训练完毕======================")
    return acc


def train_scala_double_dbn(struct, struct1, num, all):
    print("=================开始使用双DBN多尺度======================")

    x1_1, x1_2, x1_3 = struct
    x2_1, x2_2, x2_3 = struct1
    x_train_scala_split_1, x_train_scala_split_2 = splitDatas(x_train_scala, num, all)
    x_valid_scala_split_1, x_valid_scala_split_2 = splitDatas(x_valid_scala, num, all)

    train_datasets = [x_train_scala_split_1, y_train, x_valid_scala_split_1, y_valid]
    train_datasets1 = [x_train_scala_split_2, y_train, x_valid_scala_split_2, y_valid]

    train_x_dim = train_datasets[0].shape[1]
    train_y_dim = train_datasets[1].shape[1]
    train_x_dim_1 = train_datasets1[0].shape[1]
    train_y_dim_1 = train_datasets1[1].shape[1]

    struct = [train_x_dim, x1_1, x1_2, x1_3, train_y_dim]
    struct1 = [train_x_dim_1, x2_1, x2_2, x2_3, train_y_dim_1]

    acc = doubleDBN(train_datasets, train_datasets1, struct, struct1)
    print("=================使用双DBN多尺度训练完毕======================")
    return acc


def train_originalscala_vmd_single_dbn(struct):
    print("=================单DBN原始多尺度VMD分解训练======================")
    x1, x2, x3 = struct
    train_originalscala_vmd_datasets = [x_train_originalscala_vmd, y_train, x_valid_originalscala_vmd, y_valid]
    train_originalscala_vmd_x_dim = train_originalscala_vmd_datasets[0].shape[1]
    train_originalscala_vmd_y_dim = train_originalscala_vmd_datasets[1].shape[1]
    acc = singleDBN(train_originalscala_vmd_datasets,
                    [train_originalscala_vmd_x_dim, x1, x2, x3, train_originalscala_vmd_y_dim])
    print("==============单DBN原始多尺度VMD分解完毕===========================")
    return acc


def train_originalscala_vmd_double_dbn(struct, struct1, num, all):
    print("=================双DBN原始多尺度VMD分解训练======================")

    x1_1, x1_2, x1_3 = struct
    x2_1, x2_2, x2_3 = struct1
    x_train_originalscala_vmd_split_1, x_train_originalscala_vmd_split_2 = splitDatas(x_train_originalscala_vmd, num,
                                                                                      all)
    x_valid_originalscala_vmd_split_1, x_valid_originalscala_vmd_split_2 = splitDatas(x_valid_originalscala_vmd, num,
                                                                                      all)

    train_datasets = [x_train_originalscala_vmd_split_1, y_train, x_valid_originalscala_vmd_split_1, y_valid]
    train_datasets1 = [x_train_originalscala_vmd_split_2, y_train, x_valid_originalscala_vmd_split_2, y_valid]

    train_x_dim = train_datasets[0].shape[1]
    train_y_dim = train_datasets[1].shape[1]
    train_x_dim_1 = train_datasets1[0].shape[1]
    train_y_dim_1 = train_datasets1[1].shape[1]

    struct = [train_x_dim, x1_1, x1_2, x1_3, train_y_dim]
    struct1 = [train_x_dim_1, x2_1, x2_2, x2_3, train_y_dim_1]

    acc = doubleDBN(train_datasets, train_datasets1, struct, struct1)
    print("==============双DBN多尺度VMD分解完毕===========================")
    return acc


def train_scala_vmd_single_dbn(struct):
    print("=================单DBN多尺度VMD分解训练======================")
    x1, x2, x3 = struct
    train_scala_vmd_datasets = [x_train_scala_vmd, y_train, x_valid_scala_vmd, y_valid]
    train_scala_vmd_x_dim = train_scala_vmd_datasets[0].shape[1]
    train_scala_vmd_y_dim = train_scala_vmd_datasets[1].shape[1]
    acc = singleDBN(train_scala_vmd_datasets, [train_scala_vmd_x_dim, x1, x2, x3, train_scala_vmd_y_dim])
    print("==============单DBN多尺度VMD分解完毕===========================")
    return acc


def train_scala_vmd_double_dbn(struct, struct1, num, all):
    print("=================双DBN多尺度VMD分解训练======================")

    x1_1, x1_2, x1_3 = struct
    x2_1, x2_2, x2_3 = struct1
    x_train_scala_vmd_split_1, x_train_scala_vmd_split_2 = splitDatas(x_train_scala_vmd, num, all)
    x_valid_scala_vmd_split_1, x_valid_scala_vmd_split_2 = splitDatas(x_valid_scala_vmd, num, all)

    train_datasets = [x_train_scala_vmd_split_1, y_train, x_valid_scala_vmd_split_1, y_valid]
    train_datasets1 = [x_train_scala_vmd_split_2, y_train, x_valid_scala_vmd_split_2, y_valid]

    train_x_dim = train_datasets[0].shape[1]
    train_y_dim = train_datasets[1].shape[1]
    train_x_dim_1 = train_datasets1[0].shape[1]
    train_y_dim_1 = train_datasets1[1].shape[1]

    struct = [train_x_dim, x1_1, x1_2, x1_3, train_y_dim]
    struct1 = [train_x_dim_1, x2_1, x2_2, x2_3, train_y_dim_1]

    acc = doubleDBN(train_datasets, train_datasets1, struct, struct1)
    print("==============双DBN多尺度VMD分解完毕===========================")
    return acc


def usePSOAndSVMDDBN(x):
    printMemoryInfo()
    ros = train_originalscala_vmd_single_dbn(x)
    print("==============准确率===============")
    print('执行完返回的平均准确率:', ros, ' 转化成最小的值:', 1 - ros)
    print("==================================")
    return 1 - ros


def startPSO():
    start = time.time()
    pso = PSO(func=usePSOAndSVMDDBN, n_dim=3, pop=20, max_iter=100, lb=[2, 2, 2], ub=[4000, 4000, 4000], w=0.8, c1=2,
              c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    end = time.time()
    print("=================结束时间================")
    time_stamp = datetime.datetime.now()
    print("结束时间", time_stamp)
    print("总共耗时:", end - start)
    print("=======================================")


def usePSOAndMSVMDDDBN(x):
    printMemoryInfo()
    x1_1, x1_2, x1_3, x2_1, x2_2, x2_3 = x
    ros = train_originalscala_vmd_double_dbn([x1_1, x1_2, x1_3], [x2_1, x2_2, x2_3], 1, 7)
    print("==============准确率===============")
    print('执行完返回的平均准确率:', ros, ' 转化成最小的值:', 1 - ros)
    print("==================================")
    return 1 - ros


def startMSVMDPSODBN():
    start = time.time()
    pso = PSO(func=usePSOAndSVMDDBN, n_dim=6, pop=20, max_iter=40, lb=[2, 2, 2], ub=[500, 500, 500], w=0.8, c1=2,
              c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    end = time.time()
    print("=================结束时间================")
    time_stamp = datetime.datetime.now()
    print("结束时间", time_stamp)
    print("总共耗时:", end - start)
    print("=======================================")


def printAllDBN(singleStruct, doubleStruct, num, all):
    train_acc = train_single_dbn(singleStruct)
    train_vmd_acc = train_vmd_single_dbn(singleStruct)
    train_originalscala_acc = train_originalscala_single_dbn(singleStruct)
    train_scala_acc = train_scala_single_dbn(singleStruct)
    train_originalscala_vmd_acc = train_originalscala_vmd_single_dbn(singleStruct)
    train_scala_vmd_acc = train_scala_vmd_single_dbn(singleStruct)

    train_acc_double = train_double_dbn(singleStruct, doubleStruct, num, all)
    train_vmd_acc_double = train_vmd_double_dbn(singleStruct, doubleStruct, num, all)
    train_originalscala_double = train_originalscala_double_dbn(singleStruct, doubleStruct, num, all)
    train_scala_acc_double = train_scala_double_dbn(singleStruct, doubleStruct, num, all)
    train_originalscala_vmd_double = train_originalscala_vmd_double_dbn(singleStruct, doubleStruct, num, all)
    train_scala_vmd_acc_double = train_scala_vmd_double_dbn(singleStruct, doubleStruct, num, all)

    print("================检测结果=====================")
    print("使用单DBN 原始数据的准确率为:", train_acc)
    print("使用单DBN VMD分解的准确率为:", train_vmd_acc)
    print("使用单DBN 原始多尺度的准确率为:", train_originalscala_acc)
    print("使用单DBN 原始多尺度VMD分解的准确率为:", train_originalscala_vmd_acc)
    print("使用单DBN 改进多尺度的准确率为:", train_scala_acc)
    print("使用单DBN 改进多尺度VMD分解的准确率为:", train_scala_vmd_acc)

    print("使用双DBN 原始数据的准确率为:", train_acc_double)
    print("使用双DBN VMD分解的准确率为:", train_vmd_acc_double)
    print("使用双DBN 原始多尺度的准确率为:", train_originalscala_double)
    print("使用双DBN 原始多尺度VMD分解的准确率为:", train_originalscala_vmd_double)
    print("使用双DBN 改进多尺度的准确率为:", train_scala_acc_double)
    print("使用双DBN 改进多尺度VMD分解的准确率为:", train_scala_vmd_acc_double)
    print("============================================")


def AllDBNWriteLog(singleStruct, doubleStruct, num, all, VMD_nums, scala_step, split,dict):
    train_acc = train_single_dbn(singleStruct,dict)
    train_vmd_acc = train_vmd_single_dbn(singleStruct,dict)
    train_originalscala_acc = train_originalscala_single_dbn(singleStruct,dict)
    train_scala_acc = train_scala_single_dbn(singleStruct,dict)
    train_originalscala_vmd_acc = train_originalscala_vmd_single_dbn(singleStruct,dict)
    train_scala_vmd_acc = train_scala_vmd_single_dbn(singleStruct,dict)

    train_acc_double = train_double_dbn(singleStruct, doubleStruct, num, all,dict)
    train_vmd_acc_double = train_vmd_double_dbn(singleStruct, doubleStruct, num, all,dict)
    train_originalscala_double = train_originalscala_double_dbn(singleStruct, doubleStruct, num, all,dict)
    train_scala_acc_double = train_scala_double_dbn(singleStruct, doubleStruct, num, all,dict)
    train_originalscala_vmd_double = train_originalscala_vmd_double_dbn(singleStruct, doubleStruct, num, all,dict)
    train_scala_vmd_acc_double = train_scala_vmd_double_dbn(singleStruct, doubleStruct, num, all,dict)
    str1 = "================检测结果=====================" + "\n"
    str1 = str1 + "VMD_nums = " + str(VMD_nums) + "\n"
    str1 = str1 + "scala_step = " + str(scala_step) + "\n"
    str1 = str1 + "split = " + str(split) + "\n"
    str1 = str1 + "==================================================" + "\n"
    str1 = str1 + "使用单DBN 原始数据的准确率为:" + str(train_acc) + "\n"
    str1 = str1 + "使用单DBN VMD分解的准确率为:" + str(train_vmd_acc) + "\n"
    str1 = str1 + "使用单DBN 原始多尺度的准确率为:" + str(train_originalscala_acc) + "\n"
    str1 = str1 + "使用单DBN 原始多尺度VMD分解的准确率为:" + str(train_originalscala_vmd_acc) + "\n"
    str1 = str1 + "使用单DBN 改进多尺度的准确率为:" + str(train_scala_acc) + "\n"
    str1 = str1 + "使用单DBN 改进多尺度VMD分解的准确率为:" + str(train_scala_vmd_acc) + "\n"
    str1 = str1 + "使用双DBN 原始数据的准确率为:" + str(train_acc_double) + "\n"
    str1 = str1 + "使用双DBN VMD分解的准确率为:" + str(train_vmd_acc_double) + "\n"
    str1 = str1 + "使用双DBN 原始多尺度的准确率为:" + str(train_originalscala_double) + "\n"
    str1 = str1 + "使用双DBN 原始多尺度VMD分解的准确率为:" + str(train_originalscala_vmd_double) + "\n"
    str1 = str1 + "使用双DBN 改进多尺度的准确率为:" + str(train_scala_acc_double) + "\n"
    str1 = str1 + "使用双DBN 改进多尺度VMD分解的准确率为:" + str(train_scala_vmd_acc_double) + "\n"
    str1 = str1 + "============================================" + "\n"
    str1 = str1 + "\n\n\n"

    # 写入到日志文件当中
    base_dir = os.getcwd()
    # 获取当前文件夹的绝对路径
    file_name = os.path.join(base_dir, '', 'log.txt')
    my_open = open(file_name, 'a')
    # 打开fie_name路径下的my_infor.txt文件,采用追加模式
    # 若文件不存在,创建，若存在，追加
    print(str1)
    print("===============正在写入文件当中==================")
    my_open.write(str1)
    print("===============写入成功=============================")
    my_open.close()


# =======================设置参数==============================
VMD_nums = 7
scala_step = 6
split = 2
struct = [320, 234, 119]
struct1 = [234, 32, 43]


# ===========================================================
path = '../vmd/cache/cache' + str(VMD_nums) + str(scala_step)
x_train, x_valid, x_train_vmd, x_valid_vmd, x_train_originalscala, x_valid_originalscala, x_train_scala, x_valid_scala, x_train_originalscala_vmd, x_valid_originalscala_vmd, x_train_scala_vmd, x_valid_scala_vmd, y_train, y_valid = preDataByVMD.preprocessByVMD(
    path=path)

x_train = x_train.astype(np.float32)
x_valid = x_valid.astype(np.float32)


y_train = y_train.astype(np.float32)
y_valid = y_valid.astype(np.float32)
x_train_vmd = x_train_vmd.astype(np.float32)
x_valid_vmd = x_valid_vmd.astype(np.float32)


x_train_originalscala = x_train_originalscala.astype(np.float32)
x_valid_originalscala = x_valid_originalscala.astype(np.float32)


x_train_scala = x_train_scala.astype(np.float32)
x_valid_scala = x_valid_scala.astype(np.float32)


x_train_originalscala_vmd = x_train_originalscala_vmd.astype(np.float32)
x_valid_originalscala_vmd = x_valid_originalscala_vmd.astype(np.float32)


x_train_scala_vmd = x_train_scala_vmd.astype(np.float32)
x_valid_scala_vmd = x_valid_scala_vmd.astype(np.float32)







# 执行
# printAllDBN(struct, struct1, split, VMD_nums)


# train_originalscala_vmd_double_dbn(struct, struct1, split, VMD_nums)

def allWrite():
    VMD_nums = 2
    scala_step = 1
    struct = [320, 234, 119]
    struct1 = [234, 32, 43]
    while scala_step <= 10:
        split = 1
        while split < VMD_nums:
            print("====================正在执行", VMD_nums, scala_step, " split=", split, "================")
            AllDBNWriteLog(struct, struct1, split, VMD_nums, VMD_nums, scala_step, split,dict)
            print("=================================================================================")
            split = split + 1
        scala_step = scala_step + 1


allWrite()
