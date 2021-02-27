#-*- coding:UTF-8 -*-
#hw1: prediction PM2.5
#target: 10 hours of PM2.5 prediction by the first 9 hours of 18 features (including PM2.5)

import sys
import pandas as pd
import numpy as np
import math

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

#4.划分训练数据为训练集和验证集 split training data into 'training set' and 'validation set'
x_train_set = x[:math.floor(len(x)*0.8), :]
y_train_set = y[:math.floor(len(x)*0.8), :]
x_validation_set = x[math.floor(len(x)*0.8):, :]
y_validation_set = y[math.floor(len(x)*0.8):, :]

print('the length of x_train, y_train, x_valid, y_valid is: ', str(len(x_train_set)), str(len(y_train_set)), str(len(x_validation_set)), str(len(y_validation_set)))

#
