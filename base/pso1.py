import numpy as np
import random
class PSO(RNN_train):
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.space = [0,10]  # ����������Χ
        self.pN = pN  #��������
        self.dim = dim  #����ά��
        self.max_iter = max_iter  #��������
        self.X = np.zeros((self.pN, self.dim))  #�������ӵ�λ�ú��ٶ�
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  #���徭�������λ�ú�ȫ�����λ��
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  #ÿ���������ʷ�����Ӧֵ
        self.fit = 1e10     #ȫ�������Ӧֵ
        self.loss50 = 1e10


    def init_Population(self):
        #��ʼ����Ⱥ

        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(self.space[0], self.space[1])
                #self.V[i][j] = random.uniform(self.space[0], self.space[1])
            print("Point %d ��ʼ������ �� "%i)
            print(self.X[i]) #  ��ӡ��ʼ������
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
            print(" ��ʼ���� ȫ����ѵ�����꣺ ")
            print(self.gbest)
            print("---------\n")

    def iterator(self):
        #��������λ��----------------------------------
        print("--------------------------------------\n")
        print("--------------------------------------\n")
        print("��������λ��")
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
                if (temp < self.p_fit[i]):  #���¸�������
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):  #����ȫ������
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
                        self.loss50 = peak_half_loss_ave
            fitness.append(self.fit)
            best_pos.append(self.gbest)
            loss50_list.append(self.loss50)
            print("-------------------\n")
            print("�� %d ��ѭ������� "%t)
            print("ȫ�����ŵ����� :")
            print(self.gbest)
            print("MSE: %f"%self.fit)  #�������ֵ
            print(self.loss50)
            print("--------------------------------------\n")
            print("--------------------------------------\n")
        return fitness,MSE,best_pos
if __name__ == '__main__':
    my_pso = PSO(pN=5,dim=2,max_iter=5)
    my_pso.init_Population()
    fitness,best_pos = my_pso.iterator()