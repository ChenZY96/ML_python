import pandas as pd
import numpy as np
import os
import operator

def img2vec(filename):
    returnVec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j] = int(lineStr[j])
    return  returnVec

def setDataSet(trainpath,testpath):
    trainFile = os.listdir(trainpath)
    train_num = len(trainFile)
    trainLabel = []
    trainData = np.zeros((train_num,1024))
    for i in range(train_num):
        number = trainFile[i].split('_')[0]
        trainLabel.append(int(number))
        trainVec = img2vec('digits/trainingDigits/'+trainFile[i])
        trainData[i,:] = trainVec
    testFile = os.listdir(testpath)
    test_num = len(testFile)
    testLabel = []
    testData = np.zeros((test_num,1024))
    for j in range(test_num):
        number = testFile[j].split('_')[0]
        testLabel.append(int(number))
        testVec = img2vec('digits/testDigits/'+testFile[j])
        testData[j,:] = testVec
    return trainData,trainLabel,testData,testLabel

def classify0(inX,dataSet,label,k):
    size_dataSet = dataSet.shape[0]
    sqDistance = ((inX - dataSet) ** 2).sum(1)
    distance = sqDistance ** 0.5
    sortedDistance = distance.argsort()
    ClassCount = {}
    for i in range(k):
        voteILabel = label[sortedDistance[i]]
        ClassCount[voteILabel] = ClassCount.get(voteILabel,0) + 1
    result = sorted(ClassCount.items(),key = operator.itemgetter(1),reverse = True)
    return result[0][0]

def handwritingClassTest(trainData,trainLabel,testData,testLabel):
    total_test = len(testData)
    errorCount = 0
    for i in range(total_test):
        classResult = classify0(testData[i],trainData,trainLabel,3)
        print('预测结果是:{0};实际结果是:{1}'.format(classResult,testLabel[i]))
        if classResult != testLabel[i]:
            errorCount += 1
    print('错误率是:{0}'.format(errorCount/total_test))




if __name__ == '__main__':
    trainpath = 'digits/trainingDigits'
    testpath = 'digits/testDigits'

    trainData,trainLabel,testData,testLabel = setDataSet(trainpath,testpath)

    handwritingClassTest(trainData,trainLabel,testData,testLabel)


