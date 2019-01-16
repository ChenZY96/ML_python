# -*- coding: UTF-8 -*-
from math import log
import operator
import treePlotter
import pickle

def creatDataSets(filename):
    f = open(filename)
    dataSet = [each.strip().split('\t') for each in f.readlines()]
    labels = ['年龄','症状','是否散光','眼泪','分类标签']
    return dataSet,labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt
"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    feat_axis - 划分数据集的特征索引
    value - 需要返回的特征的值
Returns:
    retDataSet
"""
def splitDataSet(dataSet,feat_axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[feat_axis] == value:
            reduceFeatVec = featVec[:feat_axis]
            reduceFeatVec.extend(featVec[feat_axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

# 计算各个特征的信息增益，选择最大的作为划分特征
def bestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 特征数量
    baseEntropy = calcShannonEnt(dataSet)
    infoGain_dict = {}
    for i in range(numFeatures):
        newEntropy = 0.0
        # 获取该特征的所有特征值
        featVals = set([each[i] for each in dataSet])
        for value in featVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        #print('第{0}个特征的信息增益是:{1}'.format(i,infoGain))
        infoGain_dict[i] = infoGain #存储信息增益
    sortedInfoGain = sorted(infoGain_dict.items(),key=operator.itemgetter(1),reverse = True)
    bestFeat  =sortedInfoGain[0][0]
    #print('最好的特征是第{0}个'.format(bestFeat))
    return bestFeat

def majorityCnt(classList):
    classCount = {}
    for each in classList:
        if each not in classCount.keys():
            classCount[each] = 0
        classCount[each] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,label,featLabel):
    classList = [each[-1] for each in dataSet] #取出所有的分类标签
    if classList.count(classList[0]) ==len(classList): # 如果所有都是同一类，则停止划分
        return classList[0]
    if len(dataSet[0]) == 1: # 遍历完所有特征后返回出现次数最多的分类标签
        return majorityCnt(classList)
    bestFeat = bestFeature(dataSet) # 选择最优特征
    bestFeatLabel = label[bestFeat]  # 最优特征的标签
    featLabel.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}} # 根据最优特征的标签生成树
    del(label[bestFeat]) # 删除已经使用特征标签
    featValues = set([each[bestFeat] for each in dataSet]) # 该最优特征的所有特征值
    for each in featValues: # 遍历特征，创建决策树
        subLabel = label[:]
        myTree[bestFeatLabel][each] = createTree(splitDataSet(dataSet,bestFeat,each),subLabel,featLabel)
    return myTree

"""
函数说明:使用决策树分类

Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""
def classify(inputTree,featLabel,testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabel.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabel,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    with open(filename,'wb') as f:
        pickle.dump(inputTree,f)
def grabTree(filename):
    f = open(filename,'rb')
    return pickle.load(f)

if __name__ == '__main__':
    filename = 'lenses.txt'
    dataSet,feature = creatDataSets(filename)
    featLabel = []
    myTree = createTree(dataSet,feature,featLabel)
    # 决策树可视化
    #treePlotter.createPlot(myTree)
    # 测试
    testVec = ['normal','yes','hyper','presbyopic','presbyopic','hyper']
    result = classify(myTree,featLabel,testVec)
    print(result)
