import numpy as np
import random
class PSO(RNN_train):
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.space = [0,10]  # 粒子搜索范围
        self.pN = pN  #粒子数量
        self.dim = dim  #搜索维度
        self.max_iter = max_iter  #迭代次数
        self.X = np.zeros((self.pN, self.dim))  #所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  #个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  #每个个体的历史最佳适应值
        self.fit = 1e10     #全局最佳适应值
        self.loss50 = 1e10


    def init_Population(self):
        #初始化种群

        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(self.space[0], self.space[1])
                #self.V[i][j] = random.uniform(self.space[0], self.space[1])
            print("Point %d 初始化坐标 ： "%i)
            print(self.X[i]) #  打印初始点坐标
            self.pbest[i] = self.X[i]
            RNN = RNN_train(self.X[i][0], self.X[i][1])
            history, model = RNN.model_train()
            RNN.train_plot(history)
            peak_shift_MSE, test_loss_total_ave, peak_half_loss_ave, peak_twenty_percent_loss_ave = RNN.loss_cal(
                model)
            self.p_fit[i] = peak_shift_MSE
            if (peak_shift_MSE < self.fit):
                self.fit = peak_shift_MSE
                self.gbest = self.X[i]
                self.loss50 = peak_half_loss_ave
            print(" 初始化中 全局最佳点的坐标： ")
            print(self.gbest)
            print("---------\n")

    def iterator(self):
        #更新粒子位置----------------------------------
        print("--------------------------------------\n")
        print("--------------------------------------\n")
        print("更新粒子位置")
        fitness = []
        best_pos = []
        loss50_list = []
        for t in range(self.max_iter):
            print("It is the %d iteration "%t)
            for i in range(self.pN):
                self.X[i] = self.w*self.X[i] + self.c1*random.random()*(self.pbest[i] - self.X[i]) + \
                            self.c2*random.random()*(self.gbest - self.X[i])
                print("Iterator : %d , Ponit : %d"%(t,i))
                print(self.X[i])
                print("--------------------------------------\n")
                RNN = RNN_train(self.X[i][0], self.X[i][1])
                history, model = RNN.model_train()
                RNN.train_plot(history)
                peak_shift_MSE, test_loss_total_ave, peak_half_loss_ave, peak_twenty_percent_loss_ave = RNN.loss_cal(
                    model)
                temp = peak_shift_MSE
                if (temp < self.p_fit[i]):  #更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):  #更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
                        self.loss50 = peak_half_loss_ave
            fitness.append(self.fit)
            best_pos.append(self.gbest)
            loss50_list.append(self.loss50)
            print("-------------------\n")
            print("第 %d 个循环结果： "%t)
            print("全局最优点坐标 :")
            print(self.gbest)
            print("MSE: %f"%self.fit)  #输出最优值
            print(self.loss50)
            print("--------------------------------------\n")
            print("--------------------------------------\n")
        return fitness,MSE,best_pos
if __name__ == '__main__':
    my_pso = PSO(pN=5,dim=2,max_iter=5)
    my_pso.init_Population()
    fitness,best_pos = my_pso.iterator()