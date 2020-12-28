import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def distEclud(dataSet, centroids):
    res = []
    for i in range(centroids.shape[0]):
        temp = []
        for j in range(dataSet.shape[0]):
            a = np.sqrt((dataSet[j] - centroids[i]) ** 2)
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
        dataSet = np.array(dataSet)
        dist = distEclud(dataSet, centroids)
        dist = torch.Tensor(dist)
        label = dist.argmin(0)
        label = label.unsqueeze(2)
        cmp_labels = label.expand(n, m, 3)
        oldCentroids = centroids.copy()
        temp = []
        dataSet = torch.Tensor(dataSet)
        for i in range(k):
            a = torch.where(cmp_labels == i, dataSet.type(torch.DoubleTensor), 0.0)
            centroidsT = torch.Tensor(centroids)
            temp.append(torch.where(a != 0, centroidsT[i].type(torch.DoubleTensor), 0.0))
            num = 0
            den = 0
            for j in range(n):
                num += a[j].sum(axis=0)
                exist = (a[j] != 0)
                den += exist.sum(axis=0)
            centroids[i][0] = num[0] / den[0]
            centroids[i][1] = num[1] / den[1]
            centroids[i][2] = num[2] / den[2]
        print("第{:.0f}轮：{:.2f}".format(court, time.time() - start))

        if (oldCentroids == centroids).all():
            break
    res = torch.zeros(n, m, 3)
    for i in range(k):
        res += temp[i]
    return res.int()


start = time.time()
data = mpimg.imread('1.jpeg')
img = kMeans(data, 2)

plt.imshow(img)
plt.show()
print("总耗时：{:.2f}".format(-start + time.time()))
