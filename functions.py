import itertools
import matplotlib.pyplot as plt
from random import uniform, seed
from scipy.spatial import distance
from utils import most_common, ratio, calcMeanByCoord, loadData
from sklearn.metrics import silhouette_score
import numpy as np

#compare prediction labels with true labels
def measureAccuracy(X, Cindex, n, numberOfClasses):
    classTrue = [0] * n
    classPred = [0] * n

    #get true labels
    for j in range(0, n):
        classTrue[j] = X[j]['class']

    #pick a label for each class that gives the best results
    Cindex.sort(key=len)
    usedLabels = []
    for j in range(0, numberOfClasses):
        temp = []
        for i in range(0, len(Cindex[j])):
            if not X[Cindex[j][i]]['class'] in usedLabels:
                temp.append(X[Cindex[j][i]]['class'])

        for i in range(0, len(Cindex[j])):
            classPred[Cindex[j][i]] = most_common(temp)
            usedLabels.append(most_common(temp))

    #calc presision and recall values
    result = ratio(classTrue, classPred)
    print("Precision:", result[0])
    print("Recall:", result[1])

    return classPred

# k means algorithm
def kmeans(fname, dim, K, draw):
    seed(5)

    X = loadData(fname)
    n = len(X)

    C = [[] for i in range(K)]
    D = [[] for i in range(K)]
    Cindex = [[] for i in range(K)]
    Dindex = [[] for i in range(K)]

    m = []
    dist = [0] * K

    #get random centroids
    for i in range(0, K):
        t = []
        for d in range(0, dim):
            t.append(round(uniform(-3.0, 3.0), 2))
        m.append(t)


    while True:
        for j in range(0, n):
            for k in range(0, K):
                dist[k] = distance.euclidean((X[j]['x'], X[j]['y']), m[k])
            D[dist.index(min(dist))].append([X[j]['x'], X[j]['y']])
            Dindex[dist.index(min(dist))].append(j)

        for i in range(0, K):
            if D[i] == C[i]:
                break
        else:
            for i in range(0, K):
                m[i] = calcMeanByCoord(D[i])
                C[i] = list(D[i])
                Cindex[i] = Dindex[i]
                D[i][:] = []
        break

    #draw plot
    if draw:
        colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        for i in range(0, K):
            plt.scatter(*zip(*C[i]), color=next(colors))
    plt.show()

    #measure accuracy
    out = measureAccuracy(X, Cindex, n, K)
    return out


def findNumberOfClasses(fname, dim, classes):

    coord = []
    X = loadData(fname)
    n = len(X)

    for j in range(0, n):
        coord.append([X[j]['x'], X[j]['y']])

    coord = np.array(coord)

    for nclass in classes:
        label = kmeans(fname, dim, nclass, draw=False)

        label = np.array(label)
        print(silhouette_avg = silhouette_score(coord, label))

