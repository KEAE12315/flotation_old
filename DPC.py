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
from haversine import haversine
from utils.itertool import *

import logging
dpc_logger = logging.getLogger('dpc')


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

        self.logger = logging.getLogger('dpc.DPC')
        self.logger.info('Total number of tracks: '+str(len(self.df.trackID.unique())))
        self.logger.info('Total number of points: '+str(self.df.shape[0]))

    def loadDis(self, dis):
        """当输入数据为距离矩阵时可以直接导入不必计算
        """
        self.dis = dis
        self.logger.info('Distance matrix was imported successfully')

    def calDc(self, method='proportion', **kwargs):
        """@description  :计算截断距离dc

        Args
        - method: 计算dc的方法, 默认为proportion. 各方法所需参数需用关键词给出.
            - proportion: 按照所有点之间的距离中选取前百分之多少(p)选取dc. 关键词参数p
            - Bonferroni: 根据邦费罗尼指数选取dc
        """
        logger = logging.getLogger('dpc.DPC.calDc')
        loggerP = logging.getLogger('dpc.DPC.calDc.proportion')
        loggerB = logging.getLogger('dpc.DPC.calDc.Bonferroni')
        logger.info('Calculate dc by method '+method)

        def proportion():
            tmp = np.tril(self.dis)
            tmp = tmp.ravel()
            tmp = tmp[np.where(tmp)]
            tmp = np.sort(tmp)
            self.dc = tmp[round(len(tmp) * kwargs['p'])]

            loggerP.debug('Order of all distances: '+str(tmp))

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

        tmp = locals()[method]()
        logger.info('Cut-off distance calculated successfully as '+str(self.dc)+' with '+method)
        return tmp

    def calDis(self):
        """计算距离矩阵
        """
        logger = logging.getLogger('dpc.DPCL.calDis')

        disMatrix = np.zeros((self.N, self.N))
        for i, ix, iy in self.df[['x', 'y']].itertuples():
            for j, jx, jy in self.df.loc[i + 1:, ['x', 'y']].itertuples():
                d = norm2(ix, iy, jx, jy)
                disMatrix[i, j] = d
                disMatrix[j, i] = d

        logger.info('Distance matrix calculated successfully')

        self.dis = disMatrix
        return disMatrix

    def calRho(self, dc=None, kernel='gaussian'):
        """计算局部密度

        Args:
            - dc: 截断距离, 默认调用self.dc.
            - kernel: 计算方式
                - gaussian: 高斯核
                - cutoff: 截断式
        """
        logger = logging.getLogger('dpc.DPCL.calRho')
        logger.info('Local density calculated with method '+kernel)

        rho = np.zeros(self.N)

        if dc == None:
            dc = self.dc
        else:
            dc = dc

        def cutoff():
            if self.dis[i, j] < dc:
                return 1

        def gaussian():
            return np.exp(-(self.dis[i, j] / dc)**2)

        for i, *_ in self.df[:self.N - 2].itertuples():
            for j, *_ in self.df[i + 1:self.N - 1].itertuples():
                tmp = locals()[kernel]()
                rho[i] = rho[i] + tmp
                rho[j] = rho[j] + tmp

            logger.debug('rho of '+str(i)+': '+str(tmp))

        self.df['rho'] = rho
        return rho

    def calDel(self):
        logger = logging.getLogger('dpc.DPCL.calDel')

        delta = np.zeros(self.N)
        toh = np.zeros(self.N)

        for i, ri in self.df.iterrows():
            indexs = self.df[self.df['rho'] > ri['rho']].index.values
            if indexs.any():
                disp = pd.Series(self.dis[i, :])[indexs]
                delta[i] = disp.min()
                toh[i] = disp.idxmin()
                logger.debug(str(i)+' to high density point '+str(toh[i]))
            else:
                logger.info('Highest density point: '+str(i))
                delta[i] = max(self.dis[i, :])
                toh[i] = -1

        self.df['delta'] = delta
        self.df['toh'] = toh

        return delta, toh

    def calGam(self):
        self.logger.info('Gamma calculating')
        self.df['gamma'] = self.df['rho'] * self.df['delta']

    def getCen(self, n):
        self.centers = self.df.sort_values(by='gamma', ascending=False).index.values[:n]
        self.logger.info(str(n)+' center points: '+str(self.centers))

    def cluster(self):
        logger = logging.getLogger('dpc.DPC.cluster')

        def group(i, id):
            self.df.loc[i, 'clusterID'] = id
            logger.debug(str(i)+' was classified as cluster '+str(id))

            for j in self.df[self.df.toh == i].index.tolist():
                group(j, id)

        # 聚类中心不应该指向比它密度更高的聚类中心
        self.df.loc[self.centers, 'toh'] = -1

        for idCluster, indexCenter in enumerate(self.centers):
            group(indexCenter, idCluster + 1)
            logger.info(str(self.df[self.df.clusterID == idCluster + 1].shape[0])+'points in cluster '+str(idCluster+1) + ' with center point '+str(indexCenter))


class DPCLink(DPC):
    """针对GPS坐标聚类, 采用区域一致性指数作为距离"""

    def __init__(self, data) -> None:
        super().__init__(data)
        self.logger = logging.getLogger('dpc.DPCL')

    def calDis(self):
        """计算区域一致性矩阵及距离矩阵
        """
        logger = logging.getLogger('dpc.DPCL.calDis')
        logger.info('Distance matrix calculation begins')

        # 区域一致性矩阵
        self.dis = np.zeros((self.N, self.N))
        # 距离矩阵
        self.dis2 = np.zeros((self.N, self.N))

        for p in self.df.itertuples():
            i = p.Index

            for q in self.df[i + 1:].itertuples():
                j = q.Index

                d1 = coh(p, q)
                d2 = haversine([p.lat, p.lng], [q.lat, q.lng])
                self.dis[i, j] = d1
                self.dis[j, i] = d1
                self.dis2[i, j] = d2
                self.dis2[j, i] = d2

                # logger.debug(str(i)+' '+str(j)+' d1:'+str(d1))
                # logger.debug(str(i)+' '+str(j)+' d2:'+str(d2))
            logger.debug(str(i))

        logger.info('Distance matrix calculation ends')

    def calDel(self):
        logger = logging.getLogger('dpc.DPCL.calDel')

        delta = np.zeros(self.N)
        toh = np.zeros(self.N)

        for i, ri in self.df.iterrows():
            indexs = self.df[self.df['rho'] > ri['rho']].index.values
            if indexs.any():
                disp = pd.Series(self.dis2[i, :])[indexs]
                delta[i] = disp.min()
                toh[i] = disp.idxmin()
            else:
                logger.info('The point of maximum density: '+str(i))
                delta[i] = max(self.dis2[i, :])
                toh[i] = -1

        self.df['delta'] = delta
        self.df['toh'] = toh


class DPCLink_TE(DPCLink):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.logger = logging.getLogger('dpc.DPCL-TE')

    def TSC():
        """Temporal sequence constraint"""
        pass

    def _ei(sq, d):
        """计算一个序列的混乱指数, d为分组数"""

        # 生成分组区间
        ndNs = {}
        for l, r in Two(np.linspace(0, 360, d+1)):
            ndNs[str(l)+'-'+str(r)] = []
            ndNs[str(l)+'-'+str(r)].append(pd.Interval(left=l, right=r, closed='right'))
            ndNs[str(l)+'-'+str(r)].append(0)

        # 计算落入区间内数量
        for s in sq:
            for n in ndNs:
                if s in ndNs[n][0]:
                    ndNs[n][1] = ndNs[n][1]+1
        ndNs = [ndNs[n][1] for n in ndNs]
        ndNs = [x/sum(ndNs)for x in ndNs]

        ei = -sum([x*math.log(x, math.e) for x in ndNs if x != 0])
        return ei
