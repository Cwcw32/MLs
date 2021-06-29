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
# 将字符串转换为整数型，数据处理
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
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

def load_data():
    """:cvar
    @:param：
    @:return:
    """
    name = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data = pd.read_csv('./data/car data.csv', names=name)
    print(data)
    for i in data.columns:
        print(data[i].unique(), "\t", data[i].nunique())

    sns.countplot(x=data['class'])
    plt.show()
    le = LabelEncoder()
    for i in data.columns:
        data[i] = le.fit_transform(data[i])
    X = data[data.columns[:-1]]
    y = data['class']  # 标签
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=10)

    return X_train, X_test, y_train, y_test


def test_skl_LR(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(solver='newton-cg',multi_class='multinomial')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


if __name__ == '__main__':
    print('扩展任务')
    # 设置支持中文字体
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    print("dimensions of train: {}".format(X_train.shape))
    print("dimensions of test: {}".format(X_test.shape))
    print(X_train.describe())

 #   test_skl_LR(X_train, X_test, y_train, y_test)

    # 三七开
    # # 选择逻辑回归进行训练模型
    # logreg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    # logreg.fit(X_train, y_train)
    # LogisticRegression(multi_class='multinomial', solver='newton-cg')
    # pred = logreg.predict(X_test)
    # logreg.score(X_test, y_test)
    #
    # from sklearn.model_selection import learning_curve
    #
    # lc = learning_curve(logreg, X_train, y_train, cv=10, n_jobs=-1)
    # size = lc[0]
    # train_score = [lc[1][i].mean() for i in range(0, 5)]
    # test_score = [lc[2][i].mean() for i in range(0, 5)]
    # fig = plt.figure(figsize=(12, 8))
    # plt.plot(size, train_score)
    # plt.plot(size, test_score)
    # plt.show()
