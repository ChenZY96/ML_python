# -*- coding: UTF-8 -*-
from os import listdir
import re
import numpy as np
import random

def textParse(content,filename):
    f = open(filename,errors='ignore')
    emailWords = re.split(r'\W*',f.read())
    words = [each.lower() for each in emailWords if len(each) > 0]
    content.append(words)
def createVocabList(content):
    myVocabList = []
    for eachlist in content:
        for each in eachlist:
            myVocabList.append(each)
    return list(set(myVocabList))

def setOfWords2Vec(vocabList,inputSet):
    retVec = np.zeros(len(vocabList))
    for each in vocabList:
        if each in inputSet:
            retVec[vocabList.index(each)] = 1
        else:
            retVec[vocabList.index(each)] = 0
    return retVec

def trainNB(trainMatrix,classLabel):
    pSpam = sum(classLabel) / float(len(classLabel)) # 计算p(spam)的概率
    WSpam = np.ones(len(trainMatrix[0]))
    WHam = np.ones(len(trainMatrix[0]))
    for i in  range(len(trainMatrix)):
        if classLabel[i] == 1: # 垃圾邮件
            WSpam += trainMatrix[i]
        else:                  # 正常邮件
            WHam += trainMatrix[i]
    pWSpam = np.log(WSpam / (2+float(sum(WSpam)))) # 计算p(W0|Spam) p(W1|Spam) p(W2|Spam)...
    pWHam = np.log(WHam / (2+float(sum(WHam)))) # 计算p(W0|Ham) p(W1|Ham) p(W2|Ham)...
    return pWSpam,pWHam,pSpam
def classify(inputVec,pWSpam,pWHam,pSpam):
    classify1 = inputVec * pWSpam
    classify0 = inputVec * pWHam
    pSpamW = sum(classify1)+np.log(pSpam) # ln(A*B) =lnA+lnB
    pHamW = sum(classify0)+np.log(1-pSpam)
    if pSpamW > pHamW:
        return 1
    if pSpamW < pHamW:
        return 0
def testEmail():
    content = []
    classLabel = []
    hamfile = listdir('email/ham')
    for each in hamfile:
        classLabel.append(1)  # 正常邮件标记为1
        filename = 'email/ham/' + each
        textParse(content,filename)
    spamfile = listdir('email/spam')
    for each in spamfile:
        classLabel.append(0)  # 垃圾邮件标记为0
        filename = 'email/spam/' + each
        textParse(content,filename)
    # 随机切分训练集和测试集
    testContent = []
    testLabel = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(content)))
        testContent.append(content[randIndex])
        testLabel.append(classLabel[randIndex])
        del content[randIndex]
        del classLabel[randIndex]

    myVocabList = createVocabList(content)

    trainMat = []
    for inputSet in content:
        retVec = setOfWords2Vec(myVocabList, inputSet)
        trainMat.append(retVec)

    pWSpam, pWHam, pSpam = trainNB(trainMat,classLabel)

    # 开始测试
    testMat = []
    errorCount = 0
    #testMat.append(setOfWords2Vec(myVocabList,each) for each in testContent)
    for inputTestSet in testContent:
        retTestVec = setOfWords2Vec(myVocabList,inputTestSet)
        testMat.append(retTestVec)

    for i in range(len(testMat)):
        result = classify(testMat[i],pWSpam, pWHam, pSpam)
        if result != testLabel[i]:
            errorCount += 1
            print('错误分类的测试集:{0}'.format(testContent[i]))
    errorRate = float(errorCount)/len(testContent)
    print("错误率是:{0}".format(errorRate))

if __name__ == '__main__':
    testEmail()

# 错误率是:0.0


