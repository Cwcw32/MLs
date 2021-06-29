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
# 观察拟合现象
from sklearn.model_selection import learning_curve
# 可以直接得准确值
from sklearn.metrics import accuracy_score
# 管道流 此处参考：https://blog.csdn.net/weixin_39693971/article/details/110501405
from sklearn.pipeline import make_pipeline
import matplotlib

"""
*****************************************************
**********************画图相关函数**********************
*****************************************************
"""


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
    # 热度表
    plt.colorbar()
    # 坐标轴含义
    plt.xlabel('guess')
    plt.ylabel('fact')
    plt.title(title)
    # 显示数据的值（Numbers）
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index])


def create_meshgrid_pic(plt, predict, X, Y, step=0.01):
    """
    画分类网格（二维）
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
    画散点图（二维）
    :param plt: 画图对象
    :param title: 图像标题
    :param X:数据集
    :param y:结果
    :return:
    """
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, color='red', marker='o', label='0')
    print(X[y == 0, 0].size)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, color='blue', marker='x', label='1')
    print(X[y == 1, 0].size)

    plt.xlabel('Age')
    plt.ylabel('EstimatedSalary')
    # 把说明放在左上角，具体请参考官方文档
    plt.legend(loc=2)
    plt.title(title)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1).astype(np.float32)
    train_scores_std = np.std(train_scores, axis=1).astype(np.float32)
    test_scores_mean = np.mean(test_scores, axis=1).astype(np.float32)
    test_scores_std = np.std(test_scores, axis=1).astype(np.float32)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


"""
*****************************************************
**********************手动实现相关函数**********************
*****************************************************
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# sigmoids函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 模型
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


# 损耗计算
def cost(X, Y, theta):
    left = np.multiply(-Y, np.log(model(X, theta)))
    right = np.multiply(1 - Y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / len(X)


# 计算梯度
def gradient(X, Y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - Y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad


# 洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    Y = data[:, cols - 1:]
    return X, Y


# 停止策略
def stopCriterion(type, value, threshold):
    if type == 'Stop_iter':
        return value > threshold
    elif type == 'Stop_cost':
        return abs(value[-1] - value[-2]) < threshold
    elif type == 'Stop_grad':
        return np.linalg.norm(value) < threshold


# 梯度下降求解
def descent(data, batchSize, stopType, thresh, alpha):
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    theta = np.zeros([1, 3])
    X, Y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X, Y, theta)]
    n = data.shape[0] - batchSize

    while True:
        grad = gradient(X[k:k + batchSize], Y[k:k + batchSize], theta)
        k += batchSize
        if k > n:
            k = 0
            X, Y = shuffleData(data)
        theta = theta - alpha * grad
        costs.append(cost(X, Y, theta))
        i += 1

        if stopType == 'Stop_iter':
            value = i
        elif stopType == 'Stop_cost':
            value = costs
        elif stopType == 'Stop_grad':
            value = grad
        else:
            print('stopType 参数设置错误，请重试')
            break
        if stopCriterion(stopType, value, thresh):
            break

    return theta, i, costs, grad, time.time() - init_time


def Classification(probability, threshold):
    if probability > threshold:
        return 1
    else:
        return 0


def Test_Stop_iter(data_train, X_test, y_test):
    # thresh的大小对结果的影响
    batchSize = data_train.shape[0]
    color = ['r', 'b', 'y', 'c', 'k']
    thresh = [1, 10, 100, 200, 500]
    alpha = 0.0000001
    print('#' * 50)
    print()
    theta, iter_times, costs, grad, cost_time = \
        descent(
            data_train,
            batchSize,
            'Stop_iter', 1, alpha)
    Correct_quantity = 0
    for X, y in zip(X_test, y_test):
        if Classification(model(X, theta)[0], 0.5) == y:
            Correct_quantity += 1
    print('  %7f cost_time = %3fs'
          % (Correct_quantity / len(y_test), cost_time))
    print()
    print('#' * 50)


"""
*****************************************************
**********************主函数**********************
*****************************************************
"""
if __name__ == '__main__':
    # 设置支持中文字体
    print('基础任务')
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
                                                        test_size=0.3,
                                                        random_state=6)
    # 数据归一化处理
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 逻辑回归模型训练
    lr = LogisticRegression(penalty='l1',
                            solver='liblinear'
                            )
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
    print("训练集准确率：", ((train_right_counts * 1.0) / X_train_std.shape[0]))

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

    # 观察拟合现象
    plot_learning_curve(lr, u"学习曲线", X_train, y_train)

    # 测试集 以下同理训练集
    y_test_pred = lr.predict(X_test_std)
    print(X_test_std.shape)
    test_right_counts = 0
    for i in range(X_test_std.shape[0]):
        original_val = y_test[i]
        predict_val = lr.predict(X_test_std[i, :].reshape(1, -1))
        # print "Original:", original_val, " Predict:", predict_val, ("o" if original_val == predict_val else "x")
        if original_val == predict_val:
            test_right_counts += 1

    print("测试集准确率：", ((test_right_counts * 1.0) / X_test_std.shape[0]))
    print(accuracy_score(y_test, y_test_pred))
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
    plot_confusion_matrix(y_test,
                          y_test_pred,
                          title='测试集混淆矩阵')
    plt.show()
    plt.close()

    # 其他方法，比较
    # l2正则
    lr2 = LogisticRegression(
        penalty='l2',
        solver='liblinear'
    )
    lr2.fit(X_train_std, y_train)
    y_l2_pred = lr2.predict(X_test_std)
    print('准确率,liblinear,l2', accuracy_score(y_test, y_l2_pred))
    #
    lr3 = LogisticRegression(
        penalty='l2',
        solver='newton-cg'
    )
    lr3.fit(X_train_std, y_train)
    y_l3_pred = lr3.predict(X_test_std)
    print('准确率,newton-cg,l2', accuracy_score(y_test, y_l3_pred))
    #
    lr4 = LogisticRegression(
        penalty='l2',
        solver='lbfgs'
    )
    lr4.fit(X_train_std, y_train)
    y_l4_pred = lr4.predict(X_test_std)
    print('准确率,lbfgs,l2', accuracy_score(y_test, y_l4_pred))
    #
    lr5 = LogisticRegression(
        penalty='l2',
        solver='sag'
    )
    lr5.fit(X_train_std, y_train)
    y_l5_pred = lr5.predict(X_test_std)
    print('准确率,sag,l2', accuracy_score(y_test, y_l5_pred))

    # 手动验证

    X = X_train_std
    X = np.insert(X, 0, 1, axis=1)
    X = np.column_stack((X, y_train.reshape(-1, 1)))
    #print(X[:, :X.shape[1] - 1])
    test = np.insert(X_test_std, 0, 1, axis=1)
    Test_Stop_iter(X, test, y_test)
