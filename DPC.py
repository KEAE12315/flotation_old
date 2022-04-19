# -*- encoding: utf-8 -*-
'''
@Description:
密度峰值聚类
@Date     :2022/04/18 10:36:14
@Author      :pangliwei-bjut
@version      :1.1
'''

import numpy as np
import pandas as pd
from distance import norm2, coh


class DPC:
    """密度峰值聚类算法

    输入数据支持两种类型:
    1. 集合的二维坐标. 初始化时可以直接导入坐标, 然后需要通过调用calDis()计算距离坐标
    2. 距离矩阵. 需要通过MDS反变化得到二维坐标再导入坐标, 距离矩阵可以直接通过loadDis()载入.

    输出: dataframe数据, 含clusterID列表明该点属于哪个簇
    """

    def __init__(self, data) -> None:
        self.df = data
        self.N = self.df.shape[0]
        self.dc = None

    def loadDis(self, dis):
        """当输入数据为距离矩阵时可以直接导入不必计算
        """
        self.dis = dis

    def calDc(self, method='proportion', **kwargs):
        """@description  :计算截断距离dc

        Args
        - method: 计算dc的方法, 默认为proportion. 各方法所需参数需用关键词给出.
            - proportion: 按照所有点之间的距离中选取前百分之多少(p)选取dc. 关键词参数p
            - Bonferroni: 根据邦费罗尼指数选取dc
        """

        def proportion():
            tmp = np.tril(self.dis)
            tmp = tmp.ravel()
            tmp = tmp[np.where(tmp)]
            tmp = np.sort(tmp)
            print(tmp)
            self.dc = tmp[round(len(tmp) * kwargs['p'])]
            print('dc: ' + str(self.dc))

        def Bonferroni():
            def Bonferroni_index(x):
                """计算离散邦费罗尼指数"""
                b = 0
                N = len(x)
                tmp = sum(x)
                for i, _ in x[:-1].items():
                    p = (i + 1) / N
                    q = sum(x[:i + 1]) / tmp
                    b = b + (1 - q / p)

                b = b / (N - 1)
                print('Bonferroni' + str(b))
                return b

            dcs = []
            bfs = []
            for d in np.linspace(0.001, self.dis.max(), 1000):
                print('dc' + str(d))
                self.calRho(d)
                self.calDel()
                self.calGam()
                dcs.append(d)
                bfs.append(Bonferroni_index(self.df.gamma))

            return dcs, bfs

        return locals()[method]()

    def calDis(self):
        """计算距离矩阵
        """
        disMatrix = np.zeros((self.N, self.N))
        for i, ix, iy in self.df[['x', 'y']].itertuples():
            for j, jx, jy in self.df.loc[i + 1:, ['x', 'y']].itertuples():
                d = norm2(ix, iy, jx, jy)
                disMatrix[i, j] = d
                disMatrix[j, i] = d

        self.dis = disMatrix

    def calRho(self, dc=None, kernel='gaussian'):
        """计算局部密度

        Args:
            - dc: 截断距离, 默认调用self.dc.
            - kernel: 计算方式
                - gaussian: 高斯核
                - cutoff: 截断式
        """
        rho = np.zeros(self.N)

        if dc == None:
            dc = self.dc
        else:
            dc = dc

        if kernel == 'cutoff':
            for i, *_ in self.df[:self.N - 2].itertuples():
                for j, *_ in self.df[i + 1:self.N - 1].itertuples():
                    if self.dis[i, j] < dc:
                        rho[i] = rho[i] + 1
                        rho[j] = rho[j] + 1

        elif kernel == 'gaussian':
            for i, *_ in self.df[:self.N - 2].itertuples():
                for j, *_ in self.df[i + 1:self.N - 1].itertuples():
                    tmp = np.exp(-(self.dis[i, j] / dc)**2)
                    rho[i] = rho[i] + tmp
                    rho[j] = rho[j] + tmp

        self.df['rho'] = rho

    def calDel(self):
        delta = np.zeros(self.N)
        toh = np.zeros(self.N)

        for i, ri in self.df.iterrows():
            indexs = self.df[self.df['rho'] > ri['rho']].index.values
            if indexs.any():
                disp = pd.Series(self.dis[i, :])[indexs]
                delta[i] = disp.min()
                toh[i] = disp.idxmin()
            else:
                # print(i)
                delta[i] = max(self.dis[i, :])
                toh[i] = -1

        self.df['delta'] = delta
        self.df['toh'] = toh

    def calGam(self):
        self.df['gamma'] = self.df['rho'] * self.df['delta']

    def getCen(self, n):
        self.centers = self.df.sort_values(
            by='gamma', ascending=False).index.values[:n]
        print(self.centers)

    def cluster(self):
        def group(i, id):
            self.df.loc[i, 'clusterID'] = id
            for j in self.df[self.df.toh == i].index.tolist():
                group(j, id)

        self.df.loc[self.centers, 'toh'] = -1
        for i, c in enumerate(self.centers):
            print(i, c)
            group(c, i + 1)


class DPCLink(DPC):
    """针对GPS坐标聚类, 采用区域一致性指数作为距离"""

    def calDis(self):
        """计算距离矩阵
        """
        disMatrix = np.zeros((self.N, self.N))
        for p in (self.df.itertuples()):
            i = p.Index
            for q in (self.df[i + 1:].itertuples()):
                j = q.Index
                d = coh(p, q)
                disMatrix[i, j] = d
                disMatrix[j, i] = d

        self.dis = disMatrix

    def calRho(self, dc=None):

        rho = np.zeros(self.N)

        if dc == None:
            dc = self.dc
        else:
            dc = dc

        for i, *_ in self.df[:self.N - 1].itertuples():
            for j, *_ in self.df[i + 1:self.N].itertuples():
                if self.dis[i, j] > dc:
                    rho[i] = rho[i] + 1
                    rho[j] = rho[j] + 1

        self.df['rho'] = rho


if __name__ == "__main__":
    from load import *

    dataset_path = 'dataset/Geolife Trajectories 1.3/Data/'
    for df in LoadG(dataset_path):
        df.plot(x='lng', y='lat')
        break

    dpc = DPCLink(df)
    dpc.calDis()
    dpc.calDc(p=0.5)
    dpc.calRho()
    dpc.calDel()
    dpc.df.plot.scatter(
        x='lng',
        y='lat',
        c='rho',
        colormap='viridis',
        s=5,
        figsize=(10, 10))
