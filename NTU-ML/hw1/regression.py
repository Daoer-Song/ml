#-*- coding:UTF-8 -*-
#hw1: prediction PM2.5
#target: 10 hours of PM2.5 prediction by the first 9 hours of 18 features (including PM2.5)

import sys
import os
import pandas as pd
import numpy as np
import math
import csv

def prepare_data():
    global mean_x, std_x
    #1.数据预处理 data preprocessing
    data = pd.read_csv('./NTU-ML/hw1/train.csv', encoding='big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()

    #2.特征提取 feature extraction
    month_data = {}
    for month in range(12):
        sample = np.empty([18,480]) #不对数组进行初始化
        for day in range(20):
            sample[:, day*24 : (day+1)*24] = raw_data[18*(month*20+day):18*(month*20+day+1), :]
        month_data[month] = sample

    x = np.empty([12*471, 18*9], dtype = float)
    y = np.empty([12*471, 1], dtype = float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    break
                x[month*471 + day*24 + hour, :] = month_data[month][:, day*24+hour:day*24+hour+9].reshape(1,-1)
                y[month*471 + day*24 + hour, 0] = month_data[month][9, day*24+hour+9]

    print('the shape of x: '+str(x.shape))
    print('the shape of y: '+str(y.shape))

    #3.归一化 Normalization
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

    #截至目前为止，数据处理完毕！

def train():
    #4.划分训练数据为训练集和验证集 split training data into 'training set' and 'validation set'
    x_train_set = x[:math.floor(len(x)*0.8), :]
    y_train_set = y[:math.floor(len(x)*0.8), :]
    x_validation_set = x[math.floor(len(x)*0.8):, :]
    y_validation_set = y[math.floor(len(x)*0.8):, :]

    print('the length of x_train, y_train, x_valid, y_valid is: ', str(len(x_train_set)), str(len(y_train_set)), str(len(x_validation_set)), str(len(y_validation_set)))

    #5.训练，线性回归算法的实现 training, implement linear regression
    dim = 18 * 9 + 1
    w = np.zeros([dim,1])
    x = np.concatenate((np.ones([len(x), 1]), x), axis=1).astype(float)
    learning_rate = 100
    iter_time = 1000001
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for i in range(iter_time):
        y_predict = np.dot(x, w)
        loss = np.sqrt(np.sum(np.power(y_predict-y, 2))/len(x))
        if(i%500==0):
            print(str(i) + ': ' + str(loss))
        gradient = 2*np.dot(x.transpose(), y_predict-y)
        adagrad += gradient**2
        w -= learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight_and_bias.npy', w)

def predict():
    #6.准备测试数据 preparing testing data
    testdata = pd.read_csv('./NTU-ML/hw1/test.csv', header = None, encoding = 'big5')
    test_data = testdata.iloc[: , 2:]
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, 18*9], dtype = float)
    #对数据进行整理
    for i in range(240):
        test_x[i, :] = test_data[18*i : 18*(i+1), :].reshape(1,-1)
    print('the shape of test_x: ' + str(test_x.shape))
    #归一化操作 normalization
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j])/std_x[j]
    #给测试数据加上偏置项
    test_x = np.concatenate((np.ones([240,1]), test_x), axis=1).astype(float)

    #7.预测与结果保存 prediction and saving result
    w = np.load('./NTU-ML/hw1/weight_and_bias.npy')
    ans_y = np.dot(test_x, w)
    with open('./NTU-ML/hw1/submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_'+str(i), ans_y[i][0]]
            csv_writer.writerow(row)


if __name__ == '__main__':
    # print(sys.argv)
    if sys.argv[1] == 'train':
        prepare_data()
        train()
    elif sys.argv[1] == 'predict':
        if os.path.exists('./NTU-ML/hw1/weight_and_bias.npy'):
            prepare_data()
            predict()
        else:
            print('Do not exsit weight file, please check file path: ./NTU-ML/hw1/weight_and_bias.npy')

    else:
        print('Missing a parameter, The command line format is like, python regression.py train/predict.')