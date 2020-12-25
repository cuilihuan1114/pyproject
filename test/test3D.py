#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['FangSong']  # 可显示中文字符
plt.rcParams['axes.unicode_minus'] = False

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
confusion_matrix = np.array(
    [(99, 1, 2, 2, 0, 0, 6), (1, 98, 7, 6, 2, 1, 1), (0, 0, 86, 0, 0, 2, 0), (0, 0, 0, 86, 1, 0, 0),
     (0, 0, 0, 1, 94, 1, 0), (0, 1, 5, 1, 0, 96, 8), (0, 0, 0, 4, 3, 0, 85)], dtype=np.float64)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# ij配对，遍历矩阵迭代器
iters = np.reshape([[[i, j] for j in range(7)] for i in range(7)], (confusion_matrix.size, 2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]), fontsize=7)  # 显示对应的数字

plt.ylabel('真实类别')
plt.xlabel('预测类别')
plt.tight_layout()
plt.show()