# 常用科学计算工具
import pandas as pd
# 数据集训练集划分
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


def plot_confusion_matrix(y, y_pred, cmap=plt.cm.Blues, title='混淆矩阵'):
    """
    绘制混淆矩阵
    :param y: 真实值
    :param y_pred: 预测值
    :param cmap: 热力图的颜色
    :param title: 图像标题
    :return:
    """
    cm = confusion_matrix(y, y_pred)
    classes = list(set(y))
    classes.sort()
    plt.imshow(cm, cmap)
    indices = range(len(cm))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    # 热度显示仪
    plt.colorbar()
    # 就是坐标轴含义说明了
    plt.xlabel('guess')
    plt.ylabel('fact')
    plt.title(title)
    # 显示数据，直观些
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index])
    # 显示


def create_meshgrid_pic(plt, predict, X, Y, step=0.01):
    """
    画分类网格
    :param plt: 画图对象
    :param predict: 预测对象
    :param X:特征1
    :param Y:特征2
    :param step:图像跨度
    :return:
    """
    # 确认训练集的边界
    x_min, x_max = X[:].min() - .5, X[:].max() + .5
    y_min, y_max = Y[:].min() - .5, Y[:].max() + .5
    # 生成网络数据, xx所有网格点的x坐标,yy所有网格点的y坐标
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    # xx,yy的扁平化成一串坐标点（密密麻麻的网格点平摊开来）
    d = np.c_[xx.ravel(), yy.ravel()]
    # 对网格点进行预测
    Z = predict(d)
    # 预测完之后重新变回网格的样子，因为后面pcolormesh接受网格形式的绘图数据
    Z = Z.reshape(xx.shape)
    # class_size = np.unique(Z).size
    # classes_color = ['#FFAAAA', '#AAFFAA', '#AAAAFF'][:class_size]
    # cmap_light = ListedColormap(classes_color)
    # # 接受网络化的x,y,z
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 等位线


def sandiantu(plt, X, y, title='散点图'):
    """
    画目标标签为二分类的二维散点图
    :param plt: 画图对象
    :param title: 图像标题
    :param X:数据集
    :param y:结果
    :return:
    """
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10,color='red', marker='o', label='0')
    print(X[y == 0, 0].size)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10,color='blue', marker='x', label='1')
    print(X[y == 1, 0].size)

    plt.xlabel('Age')
    plt.ylabel('EstimatedSalary')
    # 把说明放在左上角，具体请参考官方文档
    plt.legend(loc=2)
    plt.title(title)


if __name__ == '__main__':
    # 设置支持中文字体
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 加载数据
    df = pd.read_csv('./data/Purchasing Desire.csv', header=0)

    # 取其中两个特征
    X = df[['Age', 'EstimatedSalary']].values
    y = df.loc[:, 'Purchased'].values
    #  print(X[y==0,0])
    #  print(X[y==0,1])
    labels = list(set(y))

    # 画散点图
    sandiantu(plt, X, y)
    plt.show()
    plt.close()
    # print(X)
    # print(y)

    # 划分数据集和训练集
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=6)
    # 数据归一化处理
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 逻辑回归模型训练
    lr = LogisticRegression()
    lr.fit(X_train_std, y_train)
    # print(lr.score(X_train_std,y_train))
    # print(X_train_std.shape[0])
    # print(y_train)

    '-----------------训练集结果-----------------'
    print(X_train_std.shape)
    train_right_counts = 0
    for i in range(X_train_std.shape[0]):
        # print(i)
        original_val = y_train[i]
        train_predict = lr.predict(X_train_std[i, :].reshape(1, -1))
        if original_val == train_predict:
            train_right_counts += 1
    print("训练集准确率：", ((train_right_counts * 1.0) / X_train_std.shape[0])
          )

    # 训练集画散点图
    plt.clf()
    create_meshgrid_pic(plt, lr.predict, X_train_std[:, 0], X_train_std[:, 1])
    sandiantu(plt, X_train_std, y_train, title='训练集散点图', )
    plt.show()
    plt.close()

    # 训练集画混淆矩阵
    y_train_pred = lr.predict(X_train_std)
    plot_confusion_matrix(y_train, y_train_pred, title='训练集混淆矩阵')
    plt.show()
    plt.close()

    # 测试集 以下同理训练集
    print(X_test_std.shape)
    test_right_counts = 0
    for i in range(X_test_std.shape[0]):
        original_val = y_test[i]
        predict_val = lr.predict(X_test_std[i, :].reshape(1, -1))
        # print "Original:", original_val, " Predict:", predict_val, ("o" if original_val == predict_val else "x")
        if original_val == predict_val:
            test_right_counts += 1

    print("测试集准确率：", ((test_right_counts * 1.0) / X_test_std.shape[0]))
    plt.close('all')

    # 测试集画散点图
    plt.clf()
    create_meshgrid_pic(plt,
                        lr.predict,
                        X_test_std[:, 0],
                        X_test_std[:, 1])
    sandiantu(plt,
              X_test_std,
              y_test,
              title='测试集散点图')
    plt.show()
    plt.close()

    # 测试集画混淆矩阵
    lr.fit(X_test_std, y_test)
    y_test_pred = lr.predict(X_test_std)
    plot_confusion_matrix(y_test,
                          y_test_pred,
                          title='测试集混淆矩阵')
    plt.show()
    plt.close()
