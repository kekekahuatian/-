import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def distEclud(vecA, vecB):
    res = []
    for i in range(vecB.shape[0]):
        temp = []
        for j in range(vecA.shape[0]):
            a = np.sqrt((vecA[j] - vecB[i]) ** 2)
            b = np.sum(a, axis=1)
            temp.append(b)
        res.append(np.array(temp))
    return res


def randCent(dataSet, k):
    m = dataSet.shape[0]
    n = dataSet.shape[1]

    center = []
    for l in range(k):
        i = np.random.randint(m)
        j = np.random.randint(n)
        center.append(dataSet[i][j])
    return np.array(center)


def kMeans(dataSet, k):
    m = dataSet.shape[1]
    n = dataSet.shape[0]

    centroids = randCent(dataSet, k)

    court = 0
    while True:
        start = time.time()
        court += 1
        # 计算每个点到中心点的距离
        # 前2560为所有点到第一个中心的距离
        dataSet = np.array(dataSet)
        dist = distEclud(dataSet, centroids)

        dist = torch.Tensor(dist)
        label = dist.argmin(0)
        # 找新中心
        label = label.unsqueeze(2)
        # data = torch.cat([dataSet, label], dim=2)
        cmp_labels = label.expand(n, m, 3)
        oldCentroids = centroids.copy()
        temp = []
        dataSet = torch.Tensor(dataSet)
        for i in range(k):
            a = torch.where(cmp_labels == i, dataSet.type(torch.DoubleTensor), 0.0)
            centroidsT = torch.Tensor(centroids)
            temp.append(torch.where(a != 0, centroidsT[i].type(torch.DoubleTensor), 0.0))
            # a = np.array(a)
            num = 0
            den = 0
            for j in range(n):
                num += a[j].sum(axis=0)
                exist = (a[j] != 0)
                den += exist.sum(axis=0)
            centroids[i][0] = num[0] / den[0]
            centroids[i][1] = num[1] / den[1]
            centroids[i][2] = num[2] / den[2]
        print(court)
        print(time.time() - start)
        if (oldCentroids == centroids).all():
            break
    res = torch.zeros(n, m, 3)
    for i in range(k):
        res += temp[i]
    return res.int()


data = mpimg.imread('1.jpeg')
img = kMeans(data, 20)

plt.imshow(img)
plt.show()
