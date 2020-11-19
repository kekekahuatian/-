import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


# 加载数据
def creatDataSet():
    # 生成数据集
    data = make_blobs(n_samples=400, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.6, 0.2, 0.8, 0.2])

    data = data[0]
    return data


def getLeader(list):
    Min = min(np.delete(list, -1, 0))
    for i in range(len(list)):
        if Min == list[i]:
            return i


# 欧氏距离计算
def distEclud(x, y):
    dict = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    return dict


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    point = []
    m = len(dataSet)
    while len(point) < k:
        x = np.random.randint(0, 99)
        point.append(dataSet[x])
        point.append(dataSet[x + 100])
        point.append(dataSet[x + 200])
        point.append(dataSet[x + 300])
    return point


# k均值聚类
def KMeans(dataSet, k):
    m = len(dataSet)  # 样本数目
    distance = np.zeros((m, k + 1))  # 距离矩阵 [i,k]表示属于哪堆
    # 伪随机中心点
    point = randCent(dataSet, k)
    while True:

        # 遍历所有的样本
        for i in range(m):
            for j in range(k):
                # 计算每个点到每个中心的距离
                distance[i][j] = distEclud(point[j], dataSet[i])
            distance[i][k] = getLeader(distance[i])  # 根据到中心点的距离分类
        i, j = 0, 0
        oldpoint = np.copy(point)
        for i in range(k):
            point[i][0] = 0
            point[i][1] = 0
            count = 0
            for j in range(m):
                if distance[j][k] == i:
                    point[i][0] += dataSet[j][0]
                    point[i][1] += dataSet[j][1]
                    count += 1
            point[i] /= count
        if distEclud(oldpoint[0], point[0]) < 0.05:
            return distance, point
            break


def draw(dataSet, k, point, distance):

    m=len(dataSet)
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制所有的样本
    for i in range(m):
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[int(distance[i][k])])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):

        plt.plot(point[i][ 0], point[i][ 1], mark[i])

    plt.show()


dataSet = creatDataSet()
k = 4
distance, point = KMeans(dataSet, k)

draw(dataSet, k, point, distance)
