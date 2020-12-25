# -*- coding: utf-8 -*-
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

import numpy as np
from models.double.doublerbms import DoubleDBM
import sys
# sys.path.append("../base")
from models.double.doublemodel import DoubleModel
from base.base_func import act_func, Summaries


class DoubleDBN(DoubleModel):
    def __init__(self,
                 hidden_act_func='relu',
                 output_act_func='softmax',
                 loss_func='mse',  # gauss 激活函数会自动转换为 mse 损失函数
                 struct=[784, 100, 100, 10],
                 # 第二个DBN
                 struct1=[784, 100, 100, 10],

                 lr=1e-4,
                 momentum=0.5,
                 use_for='classification',
                 bp_algorithm='adam',
                 epochs=100,
                 batch_size=32,  # 每次训练的大小
                 dropout=0.3,
                 units_type=['gauss', 'bin'],
                 rbm_lr=1e-3,
                 rbm_epochs=30,
                 cd_k=1,
                 pre_train=True):
        DoubleModel.__init__(self, 'DBN')
        print("==============DBN的结构===========")
        print(struct)
        print("==================================")
        self.output_act_func = output_act_func
        self.hidden_act_func = hidden_act_func
        self.loss_func = loss_func
        self.use_for = use_for
        self.bp_algorithm = bp_algorithm
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs

        self.struct = struct
        self.struct1 = struct1

        self.batch_size = batch_size
        self.dropout = dropout
        self.pre_train = pre_train

        self.dbm_struct = struct[:-1]
        self.dbm_struct1 = struct1[:-1]

        self.units_type = units_type
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        self.rbm_epochs = rbm_epochs

        if output_act_func == 'gauss':
            self.loss_func = 'mse'
        self.hidden_act = act_func(self.hidden_act_func)
        self.output_act = act_func(self.output_act_func)

        self.build_model()

    ###################
    #    DBN_model    #
    ###################
    def build_model(self):
        print("Start building model...")
        print('DBN:')
        print(self.__dict__)
        """
        Pre-training
        """
        if self.pre_train:  # cd_k=0时，不进行预训练，相当于一个DNN
            # 构建第一个DBM
            self.pt_model = DoubleDBM(
                units_type=self.units_type,
                dbm_struct=self.dbm_struct,
                rbm_epochs=self.rbm_epochs,
                batch_size=self.batch_size,
                cd_k=self.cd_k,
                rbm_lr=self.rbm_lr)
            # 构建第二个DBM
            self.pt_model1 = DoubleDBM(
                units_type=self.units_type,
                dbm_struct=self.dbm_struct1,
                rbm_epochs=self.rbm_epochs,
                batch_size=self.batch_size,
                cd_k=self.cd_k,
                rbm_lr=self.rbm_lr)

        """
        Fine-tuning
        """
        with tf.name_scope('DBN'):
            # feed 变量 將32已改成64
            self.input_data = tf.placeholder(tf.float32, [None, self.struct[0]])  # N等于batch_size（训练）或_num_examples（测试）
            # 第二个DBM输入值
            self.input_data1 = tf.placeholder(tf.float32,
                                              [None, self.struct1[0]])  # N等于batch_size（训练）或_num_examples（测试）

            self.label_data = tf.placeholder(tf.float32, [None, self.struct[-1]])  # N等于batch_size（训练）或_num_examples（测试）
            self.keep_prob = tf.placeholder(tf.float32)
            # 权值 变量（初始化）
            self.out_W = tf.Variable(tf.truncated_normal(shape=[self.struct[-2], self.struct[-1]],
                                                         stddev=np.sqrt(2 / (self.struct[-2] + self.struct[-1]))),
                                     name='W_out')
            self.out_b = tf.Variable(tf.constant(0.0, shape=[self.struct[-1]]), name='b_out')

            # 第二个DBM输入
            self.out_W1 = tf.Variable(tf.truncated_normal(shape=[self.struct1[-2], self.struct1[-1]],
                                                         stddev=np.sqrt(2 / (self.struct1[-2] + self.struct1[-1]))),
                                     name='W_out')
            self.out_b1 = tf.Variable(tf.constant(0.0, shape=[self.struct1[-1]]), name='b_out')

            # 构建dbn
            # 构建权值列表（dbn结构）
            self.parameter_list = list()
            self.parameter_list1 = list()
            if self.pre_train:
                for pt in self.pt_model.pt_list:
                    self.parameter_list.append([pt.W, pt.bh])

                # 第二个DBM
                for pt in self.pt_model1.pt_list:
                    self.parameter_list1.append([pt.W, pt.bh])

            else:
                for i in range(len(self.struct) - 2):
                    W = tf.Variable(tf.truncated_normal(shape=[self.struct[i], self.struct[i + 1]],
                                                        stddev=np.sqrt(2 / (self.struct[i] + self.struct[i + 1]))),
                                    name='W' + str(i + 1))
                    b = tf.Variable(tf.constant(0.0, shape=[self.struct[i + 1]]), name='b' + str(i + 1))
                    self.parameter_list.append([W, b])

            self.parameter_list.append([self.out_W, self.out_b])
            # 第二个DBM
            self.parameter_list1.append([self.out_W1, self.out_b1])

            # 构建训练步
            self.logist, self.pred = self.transform(self.input_data, self.input_data1)

            self.build_train_step()

            # ****************** 记录 ******************
            # if self.tbd:
            #     for i in range(len(self.parameter_list)):
            #         Summaries.scalars_histogram('_W' + str(i + 1), self.parameter_list[i][0])
            #         Summaries.scalars_histogram('_b' + str(i + 1), self.parameter_list[i][1])
            #     tf.summary.scalar('loss', self.loss)
            #     tf.summary.scalar('accuracy', self.accuracy)
            #     self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, self.name))
            # ******************************************

    def transform(self, data_x, data_x1):
        # 得到网络输出值
        next_data = data_x  # 这个next_data是tf变量
        for i in range(len(self.parameter_list)):
            W = self.parameter_list[i][0]
            b = self.parameter_list[i][1]

            if self.dropout > 0:
                next_data = tf.nn.dropout(next_data, self.keep_prob)

            z = tf.add(tf.matmul(next_data, W), b)
            if i == len(self.parameter_list) - 1:
                logist = z
                pred = self.output_act(z)
            else:
                next_data = self.hidden_act(z)

        # 第二个DBM
        next_data1 = data_x1  # 这个next_data是tf变量
        for i in range(len(self.parameter_list1)):
            W = self.parameter_list1[i][0]
            b = self.parameter_list1[i][1]

            if self.dropout > 0:
                next_data1 = tf.nn.dropout(next_data1, self.keep_prob)

            z1 = tf.add(tf.matmul(next_data1, W), b)
            if i == len(self.parameter_list1) - 1:
                logist = tf.add(z, z1)
                pred = self.output_act(logist)
            else:
                next_data1 = self.hidden_act(z1)

        return logist, pred