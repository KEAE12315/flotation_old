import pandas._libs.lib as lib
import numpy as np
import pandas as pd
import os
import logging


class LoadD:
    def __init__(self) -> None:
        pass

    def readLine(file_path):
        x = []
        y = []
        with open(file_path) as f:
            for l in f.readlines():
                line = l.split()
                x.append(float(line[0]))
                y.append(float(line[1]))

        x = np.array(x)
        y = np.array(y)

        return x, y

    def readPd(self, file_path):
        data = pd.read_csv(file_path, delim_whitespace=True, names=['x', 'y', 'none'])[['x', 'y']]

        return data

    def disM_line(self, file_path):
        data = pd.read_csv(file_path, names=['x', 'y', 'd'], sep=' ')[['x', 'y']]
        N = max(data.max())
        disMatrix = np.zeros((N, N), dtype=np.float64)
        with open(file_path) as f:
            for l in f.readlines():
                line = l.split()
                i = int(line[0]) - 1
                j = int(line[1]) - 1
                disMatrix[i, j] = float(line[2])
                disMatrix[j, i] = float(line[2])
        return disMatrix


class Load:
    """
    可迭代容器类，按顺序返回文件夹路径下的所有文件. 
    默认行为是返回pd.readcsv的结果. 
    如需更改读取单个文件的方式, 重载readOne函数.

    Args:
        dir_path: 要迭代的文件夹。
    Return:
        读取的文件, 默认为dataframe. 根据readOne而定. 
    """

    def __init__(self, dir_path, step=1, **kwargs):
        self.dir_path = dir_path
        self.step = step
        self.kwargs = kwargs

    def readOne(self, path):
        return pd.read_csv(path)

    def filterF(self, x):
        """
        筛选不需要的文件. 实际上是filter函数的判断函数. 
        默认行为为过滤Mac OS系统下自动生成的.DS_Store文件
        """
        if x == '.DS_Store':
            return False
        else:
            return True

    def process(self, df):
        """对读取的单个文件进行处理"""
        return df

    def __iter__(self):
        for root, dirs, files in os.walk(self.dir_path, topdown=True):
            dirs.sort()
            files.sort()
            files = filter(self.filterF, files)

            for name in files:
                df = self.readOne(os.path.join(root, name))
                df = self.process(df)
                yield df[::self.step]


class LoadG(Load):
    """读取Geolife Trajectories 1.3数据集, 默认步长12.
    """

    def __init__(self, dir_path, step=12):
        super().__init__(dir_path)
        self.step = step

    def readOne(self, path):
        df = pd.read_csv(path,
                         header=5,
                         names=['lat', 'lng', 'useless', 'alt', 'days', 'date', 'time'],
                         usecols=[0, 1, 5, 6])

        df['datetime'] = df.date + ' ' + df.time
        df.datetime = pd.to_datetime(df.datetime)
        df.drop(columns=['date', 'time'], inplace=True)
        df = df.iloc[::self.step, :]
        df = df.reset_index(drop=True)

        return df

    def filterF(self, x):
        if x == '.DS_Store':
            return False
        elif x == 'labels.txt':
            return False
        else:
            return True


class LoadXY(Load):
    """读取csv文件, 每行为x, y, label"""

    def readOne(self, path):
        return pd.read_csv(path, names=['x', 'y', 'clusterID'], sep='\t')


class LoadFeed(Load):
    """读取csv格式浮选数据集, 日期加上三输入两输出"""

    def __init__(self, dir_path, step=1, **kwargs):
        super().__init__(dir_path, step, **kwargs)

    def process(self, df):
        # 去除列名中的空格
        col_names = df.columns.tolist()  # 获取列名字
        for index, value in enumerate(col_names):
            col_names[index] = value.replace(" ", "").replace('%', '')
        col_names
        df.columns = col_names
        df = df[['StarchFlow', 'AminaFlow', 'OrePulpFlow', 'IronConcentrate', 'SilicaConcentrate']]
        return df
