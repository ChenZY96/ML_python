# -*- coding: UTF-8 -*-
import numpy as np
from functools import reduce
"""
函数说明:创建实验样本

Parameters:
    无
Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量
"""
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] # 1表示侮辱，0表示非侮辱
    return postingList,classVec

def creatVocabList(dataSet):
    myVocabList = []
    for eachList in dataSet:
        for each in eachList:
                myVocabList.append(each)

    return list(set(myVocabList))
def setOfWords2Vec(vocabList,inputSet):
    retVec = [0]*len(vocabList)
    for each in inputSet:
        if each not in vocabList:
            print('{0}不在本字典里'.format(each))
        else:
            retVec[vocabList.index(each)] = 1
    return retVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) # 计算用于训练的文档数量
    numWords = len(trainMatrix[0]) # 计算每条文档的词的数量
    pAbuse = trainCategory.count(1)/float(numTrainDocs) # 文档属于侮辱类的概率 即P(C1)
    w1Num = np.ones(numWords) # 所有词的出现次数初始化为1，拉普拉斯平滑
    w0Num = np.ones(numWords)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # for each in trainMatrix[i]:
            #     if each == 1:
            #         w1Num[i] += 1
            w1Num += trainMatrix[i] # 统计侮辱类文档中，每个词出现的次数
        else:
            w0Num += trainMatrix[i]
    # 分母初始化为2，拉普拉斯平滑
    p1 = np.log(w1Num / (2+float(sum(w1Num)))) # 计算条件概率p(w|C1)
    p0 = np.log(w0Num / (2+float(sum(w0Num)))) # 计算条件概率p(w|C0)
    return p0,p1,pAbuse

def classifyNB(VecInput,p0,p1,pAbuse):
    # classfy_p1 = reduce(lambda x,y:x*y,VecInput*p1) # 计算待分类词向量的条件概率p(w|C1),注意这里假设每个词是条件独立的
    # classfy_p0 = reduce(lambda x,y:x*y,VecInput*p0) # 计算待分类词向量的条件概率p(w|C0),注意这里假设每个词是条件独立的
    # result1 = classfy_p1 * pAbuse # 计算该词向量是侮辱类的概率 这里省略了贝叶斯公式的分母,因为对分类结果没有影响
    # result0 = classfy_p0 * (1-pAbuse) # 计算该词向量是非侮辱的概率
    classfy_p1 = sum(VecInput*p1)
    classfy_p0 = sum(VecInput*p0)
    result1 = classfy_p1 + np.log(pAbuse) # ln(a*b) = ln(a) +ln(b)
    result0 = classfy_p0 +np.log(pAbuse)
    if result1 > result0:
        return "侮辱类"
    if result1 < result0:
        return "非侮辱"

def testingNB(testEntry):
    postingList, classVec = loadDataSet()
    myVocabList = creatVocabList(postingList)
    trainMat = []
    for inputSet in postingList:
        retVec = setOfWords2Vec(myVocabList, inputSet)
        trainMat.append(retVec)
    p0, p1, pAbuse = trainNB0(trainMat, classVec)

    testRetVec = setOfWords2Vec(myVocabList,testEntry)
    result = classifyNB(testRetVec,p0,p1,pAbuse)
    return result


if __name__ == '__main__':
    testEntry = ['love', 'my', 'dalmation']
    result = testingNB(testEntry)
    print(result)

