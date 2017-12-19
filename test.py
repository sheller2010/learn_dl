# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:09:58 2016
Test MNIST dataset
@author: liudiwei
"""

from sklearn import neighbors
from data_util import DataUtils
import datetime
import matplotlib.pyplot as plt
import numpy as np


def main():
    trainfile_X = 'D:/DataSet/train-images.idx3-ubyte'
    trainfile_y = 'D:/DataSet/train-labels.idx1-ubyte'
    testfile_X = 'D:/DataSet/t10k-images.idx3-ubyte'
    testfile_y = 'D:/DataSet/t10k-labels.idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    return train_X, train_y, test_X, test_y


def testKNN():
    train_X, train_y, test_X, test_y = main()
    startTime = datetime.datetime.now()
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_X, train_y)
    match = 0
    for i in xrange(len(test_y)):
        predictLabel = knn.predict(test_X[i])[0]
        if (predictLabel == test_y[i]):
            match += 1

    endTime = datetime.datetime.now()
    print 'use time: ' + str(endTime - startTime)
    print 'error rate: ' + str(1 - (match * 1.0 / len(test_y)))


if __name__ == "__main__":
    # train_X, train_y, test_X, test_y = main()
    # print train_X
    # print train_y
    # print test_X
    # print test_y
    # img = np.array(train_X[111])
    # img = img.reshape(28, 28)
    # plt.figure()
    # plt.imshow(img, cmap='binary')  # 将图像黑白显示
    # plt.show()
    testKNN()

