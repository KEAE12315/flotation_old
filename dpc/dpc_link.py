import logging
import numpy as np
import pandas as pd
from haversine import haversine
from dpc.dpc_raw import DPC
from utils.distance import coh


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
                logger.info('The point of maximum density: ' + str(i))
                delta[i] = max(self.dis2[i, :])
                toh[i] = -1

        self.df['delta'] = delta
        self.df['toh'] = toh
