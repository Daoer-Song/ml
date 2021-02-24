# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random

"""
函数说明：读取数据

Parameters：
    fileName - 文件名
Returns：
    dataMat - 数据矩阵
    labelMat - 数据标签
"""
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


"""
函数说明：简化版SMO算法

Parameters:
    dataMatiIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
Returns：
    b - 
    alphas - 
"""
def smoSimple(dataMatiIn, classLabels, C, toler, maxIter):
    #转换为numpy的mat存储
    dataMatrix = np.mat(dataMatiIn); labelMat = np.mat(classLabels).transpose()




if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('./svm/testSe.txt')
    b,alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 50)

