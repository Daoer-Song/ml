import cv2
import numpy as np
from numba import cuda
import numba
import time
import math

def process_gpu(img, rows, cols, channels):
    pass

def process_cpu(img):
    pass

def process_cpu_numba(img):
    pass

if __name__ == '__main__':
    filename = './NTU-ML/hw1/daoer.jpg'
    img = cv2.imread(filename)
    img2 = cv2.imread(filename)

    # img = cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA)
    dImg = cuda.to_device(img)