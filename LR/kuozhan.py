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

if __name__ == '__main__':
    print('扩展任务')
    print('数据集为波士顿房价数据集：load-boston（）')
