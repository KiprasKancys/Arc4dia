from sklearn.metrics import confusion_matrix
import json

def loadData(fname):
    with open(fname) as data_file:
        return json.load(data_file)

def calcMeanByCoord(X):
    sumX, sumY = (0, 0)
    n = len(X)
    for i in range(0, n):
        sumX += X[i][0]
        sumY += X[i][1]

    return [sumX/n, sumY/n]

def ratio(classTrue, classPred):
    CM = confusion_matrix(classTrue, classPred)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall

def most_common(lst):
    return max(set(lst), key=lst.count)
