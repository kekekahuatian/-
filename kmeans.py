import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import torch


def creatDataSet():
    # 生成数据集
    data = make_blobs(n_samples=400, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]])

    data = data[0]
    return data


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power((vecA - vecB), 2)))


def randCent(dataSet, k):
    m = np.shape(dataSet)[1]

    center = np.mat(np.ones((k, m)))

    for i in range(m):
        centmin = min(dataSet[:, i])

        centmax = max(dataSet[:, i])

        center[:, i] = centmin + (centmax - centmin) * np.random.rand(k, 1)

    return center


def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]

    clusterAssment = np.mat(np.zeros((m, 2)))

    centroids = createCent(dataSet, k)

    clusterChanged = True

    while clusterChanged:

        clusterChanged = False

        for i in range(m):

            minDist = np.inf

            minIndex = -1

            for j in range(k):

                distJI = distMeans(dataSet[i, :], centroids[j, :])

                if distJI < minDist:
                    minDist = distJI

                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]

            centroids[cent, :] = np.mean(ptsInClust, axis=0)

    return centroids, clusterAssment


data = creatDataSet()

muCentroids, clusterAssing = kMeans(data, 5)

fig = plt.figure(0)

ax = fig.add_subplot(111)

ax.scatter(data[:, 0], data[:, 1], c=clusterAssing[:, 0].A)

plt.show()
