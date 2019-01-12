import numpy as np
from operator import itemgetter
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

def img2vector(filename):
    fr = open(filename)
    returnVec = np.zeros((1,1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,i*32+j] = int(lineStr[j])
    return returnVec

def setDataSet(trainpath,testpath):
    trainfile = listdir(trainpath)
    num_train = len(trainfile)
    trainData = np.zeros((num_train,1024))
    trainLabel = []
    for i in range(num_train):
        trainData[i,:] = img2vector('digits/trainingDigits/'+trainfile[i])
        trainLabel.append(int(trainfile[i].split('_')[0]))
    testfile = listdir(testpath)
    num_test = len(testfile)
    testData = np.zeros((num_test,1024))
    testLabel = []
    for j in range(num_test):
        testData[j,:] = img2vector('digits/testDigits/'+testfile[j])
        testLabel.append(int(testfile[j].split('_')[0]))
    return trainData,trainLabel,testData,testLabel

def handwritingClassTest(trainData,trainLabel,testData,testLabel):
    neigh = KNN(n_neighbors=3,algorithm='auto') # 构建KNN分类器
    neigh.fit(trainData,trainLabel)# 拟合模型

    errorCount = 0 # 统计错误数量

    num_test = len(testData)

    for i in range(num_test):
        result = neigh.predict([testData[i]])[0]
        print('预测结果是:{0};实际结果是:{1}'.format(result,testLabel[i]))
        if result != testLabel[i]:
            errorCount += 1
    print('错误率是:{0}'.format(errorCount/num_test))

if __name__ == '__main__':
    trainpath = 'digits/trainingDigits'
    testpath = 'digits/testDigits'

    trainData, trainLabel, testData, testLabel = setDataSet(trainpath, testpath)
    #print(trainData, trainLabel, testData, testLabel)

    handwritingClassTest(trainData, trainLabel, testData, testLabel)