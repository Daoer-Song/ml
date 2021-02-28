import cv2
import numpy as np
from numba import cuda
import numba
import time
import math

@cuda.jit
def process_gpu(img, rows, cols, channels):
    tx = cuda.blocksIdx * cuda.blocksDim.x + cuda.threadIdx.x
    ty = cuda.blocksIdy * cuda.blocksDim.y + cuda.threadIdx.y
    if tx < cols and ty < rows:
        for c in range(channels):
            color = img[i,j][c]*2.0 + 30
            if color > 255:
                img[i,j][c] = 255
            elif color < 0:
                img[i,j][c] = 0
            else:
                img[i,j][c] = color


def process_cpu(img):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                color = img[i,j][c]*2.0 + 30
                if color > 255:
                    img[i,j][c] = 255
                elif color < 0:
                    img[i,j][c] = 0
                else:
                    img[i,j][c] = color
    cv2.imwrite('./NTU-ML/hw1/song.jpg', img)

@numba.jit(nopython=True)
def process_cpu_numba(img):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                color = img[i,j][c]*2.0 + 30
                if color > 255:
                    img[i,j][c] = 255
                elif color < 0:
                    img[i,j][c] = 0
                else:
                    img[i,j][c] = color

if __name__ == '__main__':
    filename = './NTU-ML/hw1/daoer.jpg'
    img = cv2.imread(filename)
    img2 = cv2.imread(filename)

    # img = cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA)
    dImg = cuda.to_device(img)
    rows, cols, channels = img.shape
    
    dst_cpu = img.copy()
    dst_gpu = img2.copy()

    start_cpu = time.time()
    process_cpu(img)
    end_cpu = time.time()
    print('Cpu process time: '+str(end_cpu-start_cpu))

    start_cpu = time.time()
    process_cpu_numba(img)
    end_cpu = time.time()
    print('Cpu numba process time: '+str(end_cpu-start_cpu))

    threadsperblock = (32,32)
    blockspergrid_x = int(math.ceil(cols/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(rows/threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda.synchronize()
    start_gpu = time.time()
    process_gpu[blockspergrid, threadsperblock](dImg, rows, cols, channels)
    cuda.synchronize()
    end_gpu = tiem.time()
    print('Gpu process time: '+str(end_gpu-start_gpu))