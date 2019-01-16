# -*- coding: UTF-8 -*-
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as DTC
import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

filename = 'lenses.txt'
data = pd.read_csv(filename,sep='\t',names=['年龄','症状','是否散光','眼泪','分类'])

# 将数据序列化
le = LabelEncoder()  # 创建LabelEncoder()对象，用于序列化
for col in data.columns: # 为每一列序列化
    data[col] = le.fit_transform(data[col])

dataset = data.iloc[:,0:4].as_matrix()
label = data.iloc[:,4].as_matrix()
feature_name = ['年龄','症状','是否散光','眼泪']
labelName = ['hard','no lenses','soft']
dtc = DTC(criterion='entropy') # 建立决策树模型
dtc.fit(dataset,label)

dot_data = export_graphviz(dtc, out_file=None,
                           feature_names=feature_name,class_names = labelName,
                           filled=True,rounded=True,
                           special_characters=True
                           )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('tree.pdf')



