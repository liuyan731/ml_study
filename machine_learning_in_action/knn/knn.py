# -*- coding:utf-8 -*-

"""
@author: Yan Liu
@file: knn.py.py
@time: 2020/8/10 20:47
@desc: 
"""

import os
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat= tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMatrix = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMatrix[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMatrix, classLabelVector


def drawFigure():
    datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, dartingLabels = file2matrix('./dataset/datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(norMat[i, :], norMat[numTestVecs:m, :], dartingLabels[numTestVecs:m], 3)
        print("came back with:%d, real is:%d" % (classifierResult, dartingLabels[i]))
        if classifierResult != dartingLabels[i]:
            errorCount += 1.0
    print("error rate is %s" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large dosed']
    percentTats = float(input("玩游戏时间百分比"))
    ffMiles = float(input("飞行里程"))
    iceCream = float(input("吃冰淇淋量"))
    datingDataMat, dartingLabels = file2matrix('./dataset/datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, norMat, dartingLabels, 3)
    print("you will probably like this person" + resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('dataset/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        hwLabels.append(classNumstr)
        trainingMat[i, :] = img2vector('dataset/trainingDigits/' + fileNameStr)

    testFileList = os.listdir('dataset/testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('dataset/testDigits/' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with:%d , real is %d" % (classifierResult, classNumstr))
        if classNumstr != classifierResult:
            errorCount += 1.0
    print("the total num of error is: %d" % errorCount)
    print("error rate is: %f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    # group, labels = createDataSet()
    # print(classify0([0, 0], group, labels, 3))

    # datingDataMat, dartingLabels= file2matrix('./dataset/datingTestSet2.txt')
    # norMat, ranges, minVals = autoNorm(datingDataMat)
    # print(norMat)
    #
    # datingClassTest()
    #
    # drawFigure()

    # classifyPerson()

    handwritingClassTest()