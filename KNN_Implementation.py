#!/usr/bin/env python
#coding: UTF-8

import csv
import random
import math
import operator
'''
KNN算法分为如下几步：
数据处理：打开CSV文件获取数据，将原始数据分为测试集/训练集。
相似性度量：计算每两个数据实例之间的距离。
近邻查找：找到k个与当前数据最近的邻居。
结果反馈：从近邻实例反馈结果。
精度评估：统计预测精度。
主函数：将上述过程串起来。
'''
#加载数据集
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines) #list()用于把元组转换成列表
        for x in range(len(dataset)-1):   #len(dataset)表示数据集的总行数
            for y in range(4):    #一行有四个特征值
                dataset[x][y]=float(dataset[x][y]) #float() 函数用于将整数和字符串转换成浮点数。
            if random.random() < split:  #split=0.7  就是划分比例是70%
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

#计算 Euclidean Distance
def euclideanDistance(instance1, instance2, length): #例如instance1=（x1,y1,z1） instance2=（x2,y3,z2）length=3
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

#返回最近的K个邻域   从训练集里找出离测试实例最近的K个值。
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))  # 通过distances的第二个域dist排序。第三个参数reverse默认是false（升序
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#投票法  统计上述求出的邻居的分类
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]  # neighbors[x][-1]是每一个数据集的标签。
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True) # reverse=True降序
    return sortedVotes[0][0] # 投票  票数最高的类别

#预测（对数据集中的每个实例进行预测）
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('irisdata.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    #generate predictions
    predictions = []
    k = 3
    correct = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors) # 根据KNN算法 预测出来的类别
        predictions.append(result)
        # print('predictions: ' + repr(predictions))
        print('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        if result == testSet[x][-1]: # 如果预测的类别和真实的一致
            correct.append(x)

    accuracy = (len(correct)/float(len(testSet)))*100.0
    print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == '__main__':
    main()










