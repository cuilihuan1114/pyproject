# coding=UTF-8
import csv

import xlrd
import xlwt
from xlutils.copy import copy
import re
from matplotlib import pyplot as plt
import numpy as np


def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("xls格式表格写入数据成功！")


def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")


def read_excel_xls(path):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    allList = list()
    label = list()
    for i in range(0, worksheet.nrows):
        row = list()
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
            if i == 0:
                label.append(worksheet.cell_value(i, j))

            else:
                row.append(worksheet.cell_value(i, j))
        if len(row) > 0:
            allList.append(row)
        print()
    return label, allList


def read_text_log_to_excel(path):
    f = open(path, 'r', encoding='UTF-8')
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    allList = list()
    row = list()
    for line in lines:
        if line.find("VMD_nums") != -1:
            row.append(int(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("scala_step") != -1:
            row.append(int(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("split") != -1:
            row.append(int(re.findall(r"\d+\.?\d*", line)[0]))

        if line.find("使用单DBN 原始数据的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用单DBN VMD分解的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用单DBN 原始多尺度的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用单DBN 原始多尺度VMD分解的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用单DBN 改进多尺度的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用单DBN 改进多尺度VMD分解的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))

        if line.find("使用双DBN 原始数据的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用双DBN VMD分解的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用双DBN 原始多尺度的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用双DBN 原始多尺度VMD分解的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用双DBN 改进多尺度的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
        if line.find("使用双DBN 改进多尺度VMD分解的准确率为:") != -1:
            row.append(float(re.findall(r"\d+\.?\d*", line)[0]))
            print(row)
            allList.append(row)
            row = list()
    return allList


def doWrite(name):
    path = "E:\实验室资料\\1.毕业论文\\6.数据集\\日志\\2020-12-17.txt"
    result_xls = name
    sheet_name_xls = 'crwu数据集'
    value_title = [["VMD_nums", "scala_step", "split", "DBN", "VMD_DBN", "MScale_DBN"
                       , "MScale_VMD_DBN", "IMScaled_DBN", "IMScaled_VMD_DBN", "DDBN"
                       , "VMD_DDBN", "MScala_DDBN", "MScala_VMD_DDBN", "IMScala_DDBN",
                    "IMScala_VMD_DDBN"], ]
    write_excel_xls(result_xls, sheet_name_xls, value_title)
    allList = read_text_log_to_excel(path)
    write_excel_xls_append(result_xls, allList)


def doRead(name):
    return read_excel_xls(name)


def show3D(X, Y, Z, c):
    # 定义坐标轴
    fig4 = plt.figure()
    ax4 = plt.axes(projection='3d')
    ax4.set_xlabel("VMD_nums")
    # ax4.set_xlim(-6, 4)  # 拉开坐标轴范围显示投影
    ax4.set_ylabel("scala_step")
    ax4.set_zlabel("split")
    # 作图
    ax4.scatter(X, Y, Z, alpha=0.3, c=c, cmap='Oranges', s=30)  # 生成散点.利用c控制颜色序列,s控制大小
    # 设定显示范围
    plt.show()


def showBarGraph(allList_numpy, label, maxPosition):
    name_list = ['DBN', 'VMD_DBN', 'MSDBN', 'MSVMDDBN', 'IMSDBN', 'IMSVMDDBN']
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    maxPosition = np.where(allList_numpy[:, 12] == max(allList_numpy[:, 12]))[0]
    print(maxPosition)
    # 输入统计数据
    # dbn_list1 = [np.mean(allList_numpy[:, 3]), np.mean(allList_numpy[:, 4]), np.mean(allList_numpy[:, 5]),
    #              np.mean(allList_numpy[:, 6]), np.mean(allList_numpy[:, 7]), np.mean(allList_numpy[:, 8])]
    # ddbn_list1 = [np.mean(allList_numpy[:, 9]), np.mean(allList_numpy[:, 10]), np.mean(allList_numpy[:, 11]),
    #               np.mean(allList_numpy[:, 12]), np.mean(allList_numpy[:, 13]), np.mean(allList_numpy[:, 14])]
    dbn_list = [allList_numpy[maxPosition, 3][0], allList_numpy[maxPosition, 4][0], allList_numpy[maxPosition, 5][0],
                allList_numpy[maxPosition, 6][0], allList_numpy[maxPosition, 7][0], allList_numpy[maxPosition, 8][0]]
    ddbn_list = [allList_numpy[maxPosition, 9][0], allList_numpy[maxPosition, 10][0],
                 allList_numpy[maxPosition, 11][0],
                 allList_numpy[maxPosition, 12][0], allList_numpy[maxPosition, 13][0],
                 allList_numpy[maxPosition, 14][0]]

    bar_width = 0.3  # 条形宽度
    index_male = np.arange(len(name_list))
    index_female = index_male + bar_width

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_male, height=dbn_list, width=bar_width, color='y', label='单DBN')
    plt.bar(index_female, height=ddbn_list, width=bar_width, color='r', label='双DBN')

    plt.ylim(0.6, 1)  # 设置Y轴上下限
    plt.legend()  # 显示图例
    plt.xticks(index_male + bar_width / 2, name_list)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('准确率')  # 纵坐标轴标题
    plt.xlabel('模型')  # 纵坐标轴标题

    plt.title('不同模型准确率对比')
    plt.show()


def splitSum(label, nums):
    label_nums_dict = {}
    for i in range(len(label)):
        if label[i] in label_nums_dict.keys():
            label_nums_dict[label[i]] = (label_nums_dict[label[i]] + nums[i]) / 2
        else:
            label_nums_dict[label[i]] = nums[i]
    label_list = list()
    nums_list = list()
    for i in sorted(label_nums_dict):
        label_list.append(i)
        nums_list.append(label_nums_dict[i])
    return label_list, nums_list


def splitSumPosition(vmd_numslabel, label, nums, position):
    label_nums_dict = {}
    for i in range(len(label)):
        if vmd_numslabel[i] != position:
            continue
        if label[i] in label_nums_dict.keys():
            label_nums_dict[label[i]] = (label_nums_dict[label[i]] + nums[i]) / 2
        else:
            label_nums_dict[label[i]] = nums[i]
    label_list = list()
    nums_list = list()
    for i in sorted(label_nums_dict):
        label_list.append(i)
        nums_list.append(label_nums_dict[i])
    return label_list, nums_list


def showDoubleLine(label, nums1, nums2, label1, label2, xlabel_name):
    nums1_label, nums1_split = splitSum(label, nums1)
    nums2_label, nums2_split = splitSum(label, nums2)

    # 折线图
    plt.plot(nums2_label, nums1_split, 'o-', color='g', label=label1)  # o-:圆形
    plt.plot(nums2_label, nums2_split, 's-', color='r', label=label2)  # s-:方形
    plt.xlabel(xlabel_name)  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.show()
    return nums2_label[nums2_split.index(max(nums2_split))]


def showDoubleLinePosition(vmd_numslabel, label, nums1, nums2, label1, label2, xlabel_name, position):
    nums1_label, nums1_split = splitSumPosition(vmd_numslabel, label, nums1, position)
    nums2_label, nums2_split = splitSumPosition(vmd_numslabel, label, nums2, position)

    # 折线图
    plt.plot(nums2_label, nums1_split, 'o-', color='g', label=label1)  # o-:圆形
    plt.plot(nums2_label, nums2_split, 's-', color='r', label=label2)  # s-:方形
    plt.xlabel(xlabel_name)  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.show()
    return nums2_split.index(max(nums2_split))


def showLine(vmd_numslabel, label, nums1, label1, xlabel_name, maxPosition):
    nums1_label, nums1_split = splitSumPosition(vmd_numslabel, label, nums1, maxPosition)

    # 折线图
    plt.plot(nums1_label, nums1_split, 'o-', color='r', label=label1)  # o-:圆形
    plt.xlabel(xlabel_name)  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.show()


def show_loss_accuracy():
    # 读取csv至字典
    csvFile = open("D:\\pyproject\\saver\\loss_and_acc.csv", "r")
    reader = csv.reader(csvFile)
    # 建立空字典
    result = list()
    for item in reader:
        result.append(item)
    csvFile.close()
    result_np = np.array(result)
    result_np = result_np.astype(np.float64)

    # 创建数据
    x = np.arange(1, len(result_np) + 1, 1)
    train_loss = result_np[:, 0]
    train_accuracy = result_np[:, 1]
    valid_accuracy = result_np[:, 2]
    valid_loss = result_np[:, 3]

    # 创建figure窗口
    plt.figure(num=1, figsize=(8, 5))
    plt.subplot(121)
    # 画曲线1
    plt.plot(x, train_accuracy, color='r', label="train_accuracy")
    # 画曲线2
    plt.plot(x, valid_accuracy, color='orange', label="valid_accuracy")
    # 设置坐标轴名称
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    # 设置坐标轴刻度
    plt.legend(loc="best")  # 图例

    # 创建figure窗口
    plt.figure(num=1, figsize=(8, 5))
    plt.subplot(122)
    # 画曲线1
    plt.plot(x, train_loss, color='r', label="train_loss")
    # 画曲线2
    plt.plot(x, valid_loss, color='orange', label="valid_loss")
    # 设置坐标轴名称
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="best")  # 图例
    plt.show()


def label_distribution_image():
    # 读取csv至字典
    csvFile = open("D:\\pyproject\\saver\\label_distribution.csv", "r")
    reader = csv.reader(csvFile)
    # 建立空字典
    result = list()
    for item in reader:
        result.append(item)
    csvFile.close()

    # classes = ['滚珠007', '滚珠014', '滚珠021', '内圈007', '内圈014', '内圈021', '外圈007', '外圈014', '外圈021', '正常']
    classes = ['B07', 'B14', 'B21', 'IR07', 'IR14', 'IR21', 'OR07', 'OR14', 'OR21', 'NM']
    confusion_matrix = np.array(result, dtype=np.float64)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title('Confusion_matrix')

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(10)] for i in range(10)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j - 0.3, i + 0.1, round(confusion_matrix[i, j], 2))  # 显示对应的数字

    plt.xlabel('True label')
    plt.ylabel('Predict label')
    plt.tight_layout()
    plt.show()


name = '结果日志.xls'
doWrite(name)
label, allList = doRead(name)
allList_numpy = np.array(allList)

# # 打印拆线图 vmd_nums为变量
maxPosition = showDoubleLine(allList_numpy[:, 0], allList_numpy[:, 6], allList_numpy[:, 12], "MScale_VMD_DBN",
                             "MScale_VMD_DDBN",
                             "vmd_nums")
# 打印拆线图 scala_step为变量
showDoubleLinePosition(allList_numpy[:, 0],allList_numpy[:, 1], allList_numpy[:, 6], allList_numpy[:, 12], "MScale_VMD_DBN", "MScale_VMD_DDBN",
               "scala_step",maxPosition)
showLine(allList_numpy[:, 0], allList_numpy[:, 2], allList_numpy[:, 12], "MScale_VMD_DDBN", "split", maxPosition)
#
# 打印不同模型的柱状图
showBarGraph(allList_numpy, label, maxPosition)
#
# # 打印3维度的3-D图
show3D(allList_numpy[:, 0], allList_numpy[:, 1], allList_numpy[:, 2],
       allList_numpy[:, 12] * 100 * allList_numpy[:, 12] * 100)
# # 打印混淆矩阵
label_distribution_image()

# 打印准确率 与 损失率
show_loss_accuracy()
