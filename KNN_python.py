#!/usr/bin/env python
#coding: UTF-8
from sklearn import neighbors
from sklearn import datasets
#通过python实现KNN算法
knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris() #sklearn自带的数据集。
print(iris)

knn.fit(iris.data, iris.target)
predictedLabel = knn.predict([[0.7, 1.2, 1.6, 8.4]])
print(predictedLabel) #'target_names': array(['setosa', 'versicolor', 'virginica']