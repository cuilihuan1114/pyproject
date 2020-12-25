# -*- coding: utf-8 -*-
from utils import *
from scipy.io import loadmat,savemat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import  r2_score
import pandas as pd
# import tensorflow as tf
import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()  #关闭eager运算
tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
tf.set_random_seed(0)
np.random.seed(0)

def split_data(data,n):
    # 前n时刻的值为输入
    # 预测n+1个时刻的值
    in_=[]
    out_=[]
    N=data.shape[0]-n
    for i in range(N):
        in_.append(data[i:i+n,])
        out_.append(data[i+n,])
    in_=np.array(in_).reshape(-1,n)
    out_=np.array(out_).reshape(-1,1)
    return in_,out_

# In[]
file=pd.read_excel('附件1-数据.xlsx').iloc[:20,1:]
data=file.values.reshape(-1,).astype('float32')
n=100
din,dout=split_data(data,n)
#前70%作为训练集 后30为测试集
m=int(0.7*din.shape[0])
train_data,train_label=din[:m,:],dout[:m,:]
test_data,test_label=din[m:,:],dout[m:,:]

# 对数据归一化处理
ss_X=MinMaxScaler(feature_range=(0,1))
ss_Y=MinMaxScaler(feature_range=(0,1))
# ss_X=StandardScaler()
# ss_Y=StandardScaler()

x_train = ss_X.fit_transform(train_data)
y_train = ss_Y.fit_transform(train_label)
x_test = ss_X.transform(test_data)
y_test = ss_Y.transform(test_label)

# In[] 训练RBM

epoches=10
learning_rate=0.01
batchsize=128
momentum=0.9
penaltyL2=0.0
dropout=0.0
epoches_finetune=100
learning_rate_finetune=0.001
#structure=[30]#单隐含层，节点数为30
structure=[30,20]#双隐含层，节点数为30，20

opts = DLOption(epoches, learning_rate, batchsize, momentum, penaltyL2, dropout)
dbn = DBN(structure, opts, x_train)
dbn.train()

#  用各RBM堆栈DBN 并进行整个DBN的微调
opts = DLOption(epoches_finetune, learning_rate_finetune, batchsize, momentum, penaltyL2, dropout)
nn = NN(structure, opts, x_train, y_train, x_test, y_test,types=1)
nn.load_from_dbn(dbn)
start=time.time()
nn.train()
print('耗时:',time.time()-start)

test_pred=nn.predict(x_test)

# 对结果进行反归一化
test_pred = ss_Y.inverse_transform(test_pred)
test_label = ss_Y.inverse_transform(y_test)


plt.figure()
plt.plot(nn.train_loss,label='train loss')
plt.plot(nn.test_loss,label='test loss')
plt.legend()
plt.title('loss curve')
#plt.savefig('result/DBN loss curve.png')
plt.show()

# In[]计算各种指标
plt.figure()
plt.plot(test_pred,'-*',label='predict')
plt.plot(test_label,'-*',label='ture')
plt.legend()
#plt.savefig('result/DBN result.png')
plt.show()
#savemat('result/dbn_result.mat',{'true':test_label,'pred':test_pred})

# In[]计算各种指标
# mape
test_mape=np.mean(np.abs((test_pred-test_label)/test_label))
# rmse
test_rmse=np.sqrt(np.mean(np.square(test_pred-test_label)))
# mae
test_mae=np.mean(np.abs(test_pred-test_label))
# R2
test_r2=r2_score(test_label,test_pred)

print('DBN的mape:',test_mape,' rmse:',test_rmse,' mae:',test_mae,' R2:',test_r2)


