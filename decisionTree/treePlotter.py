from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt



"""
函数说明:获取决策树叶子结点的数目

Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
"""
def getNumLeafs(myTree):
    numLeaf = 0 # 初始化叶子
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该节点是否为字典，如果不是字典，代表该节点为叶子节点
        if type(secondDict[key]).__name__ == 'dict':
            numLeaf += getNumLeafs(secondDict[key])
        else:
            numLeaf += 1
    return numLeaf

"""
函数说明:获取决策树的层数

Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

"""
函数说明:绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
"""
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    arrow_args = dict(arrowstyle='<-') # 定义箭头格式
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc',size=14) # 设置中文字体
    # 绘制节点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',\
                            xytext=centerPt, textcoords='axes fraction',\
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 +cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 +cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va='center',ha='center',rotation=30)

def plotTree(myTree,parentPt,nodeTxt):
    decisionNode = dict(boxstyle='sawtooth',fc='0.8')
    leafNode = dict(boxstyle='round4',fc='0.8')
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 +float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff +1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'test')
    plt.show()