# -*- coding: UTF-8 -*-
import numpy as np
import operator
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines

'''
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    datingData - 特征矩阵
    datingLabel - 分类Label向量
'''
def file2pandas(filename):
	data = pd.read_csv(filename,names = ['airplane','game','ice-cream','label'],sep = '\t')
	datingData = data[['airplane','game','ice-cream']]
	datingLabel = data['label'].replace({'largeDoses':3,'smallDoses':2,'didntLike':1})
	# print(type(datingData))
	# print(type(datingLabel))
	return datingData,datingLabel

'''
函数说明:可视化数据

Parameters:
    datingData - 特征矩阵
    datingLabel - 分类Label
Returns:
    无
'''
def showfigure(datingData,datingLabel):
	font_path=r'/System/Library/Fonts/STHeiti Light.ttc'
	Font1 = FontProperties(fname=font_path,size=9)
	Font2 = FontProperties(fname=font_path,size=7)
	labelColor = []
	for i in datingLabel['label']:
		if i == 1:
			labelColor.append('blue')
		if i == 2:
			labelColor.append('yellow')
		if i == 3:
			labelColor.append('red')
	print(labelColor)
	fig = plt.figure()
	plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
	#每年获得的飞行常客里程数&玩视频游戏所耗时间的百分比
	ax1 = fig.add_subplot(221)
	ax1.scatter(datingData.iloc[:,0],datingData.iloc[:,1],c=labelColor,s=15,alpha=0.5) # 散点大小15 透明度0.5
	ax1.set_title(u'每年获得的飞行常客里程数&玩视频游戏所耗时间的百分比',FontProperties=Font1)
	ax1.set_xlabel(u'每年获得的飞行常客里程数',FontProperties=Font2)
	ax1.set_ylabel(u'玩视频游戏所耗时间的百分比',FontProperties=Font2)
	ax2 = fig.add_subplot(222)
	ax2.scatter(datingData.iloc[:,0],datingData.iloc[:,2],c=labelColor,s=15,alpha=0.5)
	ax2.set_title(u'每年获得的飞行常客里程数&每周消耗的冰淇淋公升数', FontProperties=Font1)
	ax2.set_xlabel(u'每年获得的飞行常客里程数', FontProperties=Font2)
	ax2.set_ylabel(u'每周消耗的冰淇淋公升数', FontProperties=Font2)
	ax3 = fig.add_subplot(223)
	ax3.scatter(datingData.iloc[:,1],datingData.iloc[:,2],c=labelColor,s=15,alpha=0.5)
	ax3.set_title(u'玩视频游戏所耗时间的百分比&每周消耗的冰淇淋公升数', FontProperties=Font1)
	ax3.set_xlabel(u'玩视频游戏所耗时间的百分比', FontProperties=Font2)
	ax3.set_ylabel(u'每周消耗的冰淇淋公升数', FontProperties=Font2)
	# 设置图例
	didntLike = mlines.Line2D([], [], color='blue', marker='.',markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='yellow', marker='.',markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label='largeDoses')
	# 添加图例
	ax1.legend(handles=[didntLike, smallDoses, largeDoses])
	ax2.legend(handles=[didntLike, smallDoses, largeDoses])
	ax3.legend(handles=[didntLike, smallDoses, largeDoses])

	plt.show()


	'''
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    data_norm - 归一化后的特征矩阵
    minVal - 每列特征的最小值
    maxVal - 每列特征的最大值

'''


def autoNorm(dataSet):
	data_norm = (dataSet - dataSet.min()) / (dataSet.max() - dataSet.min())
	#data_norm = (dataSet - dataSet.mean()) / (dataSet.std())
	minVal = dataSet.min()
	maxVal = dataSet.max()
	return data_norm,minVal,maxVal


def createDataSet():
    # group = np.array([[1.0,1.1],
	#                [1.0,1.0],
	#                [0,0],
	#                [0,0.1]])
	# labels = ['A','A','B','B']
	data_dict = {'a': [1.0, 1.0,0,0], 'b': [1.1, 1.0,0,1], 'c': ['A','A','B','B']}
	data = pd.DataFrame(data_dict)
	group = data[['a','b']]
	labels = data['c']
	return group,labels

'''
函数说明:kNN算法

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labels - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
'''
def classify0(inX,dataSet,labels,k):
	print(labels)
	print(dataSet)
	print(inX)
	#dataSetSize = dataSet.shape[0] # shape[0]计算样本数据集的行数
	diffMat = inX - dataSet.values
	#diffMat = np.tile(inX,(dataSetSize,1)) - dataSet # np.tile()将inX进行扩充，列向量上共重复dataSetSize次，行向量上重复1次
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(1) # 1表示行上相加，0表示按列相加
	distances = sqDistances ** 0.5
	sortedDistances = distances.argsort() # 从小到大排序，存储的是其index索引
	print(sortedDistances)
	ClassCount = {} # 记录各分类标签的次数
	for i in range(k):
		#labels = labels.values
		voteIlabel = labels[sortedDistances[i]]
		print(voteIlabel)
		ClassCount[voteIlabel] = ClassCount.get(voteIlabel,0) + 1 #dict.get(key,default=0) 获取key的value，如果不存在则取默认值
	# operator.itemgetter(1)表示按value排序，0表示按key排序
	sortedClassCount = sorted(ClassCount.items(),key = operator.itemgetter(1),reverse = True) #排序后是元组
	return sortedClassCount[0][0]
'''
函数说明:分类器测试函数

Parameters:
    无
Returns:
    errorCount - 错误率
'''
def datingClassTest():
	hoRatio = 0.1 # 20%作为测试集,80%作为训练集
	filename = 'datingTestSet.txt'
	datingData, datingLabel = file2pandas(filename)
	norm_data,minVal,maxVal = autoNorm(datingData)
	total_data = norm_data.shape[0]
	num_test_data = int(total_data * hoRatio)
	test_data = pd.DataFrame(norm_data[0:num_test_data])

	train_data = pd.DataFrame(norm_data[num_test_data:]).reset_index(drop=True)
	train_label = pd.Series(datingLabel[num_test_data:]).reset_index(drop=True)
	errorCount = 0 #分类错误计数

	for i in range(num_test_data):
		classifyResult = classify0(test_data[i:i+1].values[0],train_data,train_label,3)
		print('预测结果:{0};实际结果:{1}'.format(classifyResult,datingLabel.values[i]))
		if classifyResult != datingLabel.values[i]:
		 	errorCount += 1
	print('错误率是:{0}'.format(errorCount/num_test_data))


'''
函数说明:通过输入一个人的三维特征,进行分类输出

Parameters:
    无
Returns:
    无
'''
def classifyPerson():
	resultList = ['不喜欢','一般魅力','极具魅力']

	airmiles = float(input('每年获得的飞行常客里程数:'))
	games = float(input('玩视频游戏所耗时间的百分比:'))
	ice = float(input('每周消耗的冰淇淋公升数:'))

	filename = 'datingTestSet.txt'
	datingData, datingLabel = file2pandas(filename)

	norm_data,minVal,maxVal = autoNorm(datingData)

	inX = np.array([airmiles,games,ice])
	norm_inX = (inX-minVal)/(maxVal-minVal)
	item = list(norm_inX.items())
	df_data = pd.DataFrame({'0':[item[0][1]],'1':[item[1][1]],'2':[item[2][1]]})
	data = df_data.values[0]

	result = classify0(data,norm_data,datingLabel,3)
	print(result)
	print('这个人可能:{0}'.format(resultList[result-1]))




if __name__ == '__main__':
    classifyPerson()