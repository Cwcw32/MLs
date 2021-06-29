# 常用科学计算工具
import pandas as pd
# 数据集训练集划分
import seaborn as sns
from sklearn.model_selection import train_test_split
# 对特征进行标准化处理（特征缩放）
from sklearn.preprocessing import StandardScaler
# 画图用
import matplotlib.pyplot as plt
# 逻辑回归模型
from sklearn.linear_model import LogisticRegression
# 常用科学计算工具
import numpy as np
# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
# 支持中文格式
from pylab import mpl
# 比较不同模型
from sklearn import linear_model

"""
# 数据集说明
# | names file (C4.5 format) for car evaluation domain
# 
# | class values
# 
# unacc, acc, good, vgood
# 
# | attributes
# 
# buying:   vhigh, high, med, low.
# maint:    vhigh, high, med, low.
# doors:    2, 3, 4, 5more.
# persons:  2, 4, more.
# lug_boot: small, med, big.
# safety:   low, med, high.
"""

# 参考网站
# https://blog.csdn.net/weixin_42823019/article/details/112506287


if __name__ == '__main__':
    print('扩展任务')
    name = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data = pd.read_csv('./data/car data.csv', names=name)
    print(data)
    # 检查每种属性的种类
    for i in data.columns:
        print(data[i].unique(), "\t", data[i].nunique())

    sns.countplot(x=data['class'])
    plt.show()

