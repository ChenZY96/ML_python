# -*- coding:UTF-8 -*-
from os import listdir
import jieba
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def TextProcessing(folder_path):
    folder_list = listdir(folder_path)
    data_list = []
    class_list = []
    # mac系统忽略'.DS_Store'文件
    if '.DS_Store' in  folder_list:
        folder_list.remove('.DS_Store')
    # 遍历每个文件夹
    for folder in folder_list:
        files = listdir(folder_path+'/'+folder)
        # 遍历文件夹下的每个文件
        j = 1
        for file in files:
            if j > 100: # 每个类别下的文件不超过100个(避免有的分类下样本太多)
                break
            with open(folder_path+'/'+folder+'/'+file,encoding='utf-8',) as f:
                content = f.read()
                # jieba分词，精准模式
            word_cut = jieba.cut(content,cut_all=False) # 返回的是一个迭代器
            word_list = list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    #分割测试集和训练集
    train_data,test_data,train_class,test_class = train_test_split(data_list,class_list,test_size=0.2,random_state=0)
    # 统计词频
    all_words_list = {}
    for eachData in train_data:
        for each in eachData:
            if each not in all_words_list.keys():
                all_words_list[each] = 0
            all_words_list[each] += 1
    # 根据词频，降序排序
    sorted_all_words_list = sorted(all_words_list.items(),key=itemgetter(1),reverse=True)
    all_words,words_num = zip(*sorted_all_words_list)
    all_words_list = []
    for each in all_words:
        all_words_list.append(each)

    return all_words_list,train_data,test_data,train_class,test_class

# 加载停用词，以列表方式存放
def stop_words(stop_path):
    stopwords = []
    with open(stop_path,encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word)>0:
                stopwords.append(word)
    return stopwords

# 去除前100个高频词、去除停用词、去除数字、符号
def words_dict(all_words_list,deleteN,stopwords):
    feature_words = []
    n = 1 # 用于计数
    for i in range(deleteN,len(all_words_list)):
        if n > 1000: # 只取1000个特征
            break
        if all_words_list[i] not in stopwords and not all_words_list[i].isdigit() and 5>len(all_words_list[i])>1:
            feature_words.append(all_words_list[i])
            n += 1
    return feature_words

# 将文本特征向量化
def TextFeatures(train_data, test_data, feature_words):
    train_feature = []
    test_feature = []
    for eachTrain in train_data:
        tmpTrain = [0]*len(feature_words)
        for each in feature_words:
            if each in eachTrain:
                tmpTrain[feature_words.index(each)] = 1
            # else:
            #     tmpTrain[feature_words.index(each)] = 0
        train_feature.append(tmpTrain)
    for eachTest in test_data:
        tmpTest = [0]*len(feature_words)
        for each in feature_words:
            if each in eachTest:
                tmpTest[feature_words.index(each)] = 1
            # else:
            #     tmpTest[feature_words.index(each)] = 0
        test_feature.append(tmpTest)
    return train_feature,test_feature

# 新闻分类器
def TextClassify(train_data, test_data, train_class, test_class):
    clf = MultinomialNB()
    classify = clf.fit(train_data,train_class)
    accuracy = classify.score(test_data,test_class)
    return accuracy

def bestAccuracy(all_words_list,stopwords):
    accuracy_list = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords)
        print(feature_words)
        train_feature, test_feature = TextFeatures(train_data, test_data, feature_words)
        accuracy = TextClassify(train_feature, test_feature, train_class, test_class)
        accuracy_list.append(accuracy)
    print(accuracy_list)
    plt.figure()
    plt.plot(deleteNs, accuracy_list)
    plt.title('the Relationship between deleteN and accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    folder_path = './Sample'
    all_words_list,train_data, test_data, train_class, test_class = TextProcessing(folder_path)

    stop_path = 'stopwords_cn.txt'
    stopwords = stop_words(stop_path)
    deleteN = 180
    feature_words = words_dict(all_words_list,deleteN,stopwords)

    train_feature, test_feature = TextFeatures(train_data,test_data,feature_words)

    accuracy = TextClassify(train_feature, test_feature, train_class, test_class)
    print(accuracy)

    bestAccuracy(all_words_list,stopwords)
    # accuracy_list = []
    # deleteNs = range(0, 1000, 20)
    # for deleteN in deleteNs:
    #
    #     feature_words = words_dict(all_words_list, deleteN, stopwords)
    #     print(feature_words)
    #     train_feature, test_feature = TextFeatures(train_data, test_data, feature_words)
    #     accuracy = TextClassify(train_feature, test_feature, train_class, test_class)
    #     accuracy_list.append(accuracy)
    # print(accuracy_list)
    # plt.figure()
    # plt.plot(deleteNs, accuracy_list)
    # plt.title('deleteN和accuracy的关系')
    # plt.xlabel('deleteNs')
    # plt.ylabel('accuracy')
    # plt.show()


    #bestAccuracy(all_words_list,stopwords)