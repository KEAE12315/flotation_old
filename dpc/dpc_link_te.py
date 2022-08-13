import logging
import math
import numpy as np
import pandas as pd
from dpc.dpc_link import DPCLink
from utils.itertool import Two, FourDf, Df2T


def tempory(df):
    """时序约束, 将输入的df按时序切割, 并返回每段切片的起始/终点索引值. 会排除噪声, 所以切片的整体大小可能会小于原始输入.

    Arg:
        df: dataframe格式, 必须包含clusterID列. 即df已经被聚类过

    Return:
        list of list: [[切片1起始索引值, 切片1终点索引值], ...]
    """

    tmp = df[df.clusterID != -1]
    pieces = []
    for _, dfc in tmp.groupby('clusterID'):
        start = int(dfc.index[0])

        for (index_pre, index_nex), (track_pre, track_nex) in FourDf(dfc):
            if index_nex == None:
                pieces.append([start, index_pre])
            elif index_nex == index_pre + 1 and track_pre == track_nex:
                continue
            else:
                pieces.append([start, index_pre])
                start = index_nex

    # 有时会有全是噪声的情况, 切不出来片, 此时start没有被赋值, 无法引用. 但是直接过掉就好, 会返回空列表.
    # try:
    #     pieces.append([start, index_pre])
    # except UnboundLocalError:
    #     pass

    return pieces


def entropyIndex(sq: list, d: int) -> float:
    """计算一个方位角序列的混乱指数, d为分组数. 区间为0:360, 左开右闭"""

    # 生成分组区间
    ndNs = {}
    for l, r in Two(np.linspace(0, 360, d + 1)):
        ndNs[str(l) + '-' + str(r)] = []
        ndNs[str(l) + '-' + str(r)].append(pd.Interval(left=l, right=r, closed='right'))
        ndNs[str(l) + '-' + str(r)].append(0)

    # 由于不是循环区间，0无法被认为落到(:360]的区间. 所以要把0转换成360
    sq = [s if s else 360 for s in sq]

    # 计算落入区间内数量
    for s in sq:
        for n in ndNs:
            if s in ndNs[n][0]:
                ndNs[n][1] = ndNs[n][1] + 1
    ndNs = [ndNs[n][1] for n in ndNs]
    ndNs = [x / sum(ndNs) for x in ndNs]

    ei = -sum([x * math.log(x, math.e) for x in ndNs if x != 0])
    return ei


def eiPicecs(df: pd.DataFrame, thresholdEI: float, groups: int) -> bool:
    """判断轨迹序列是否满足混乱指数约束, 满足返回True

    Args:
        - df: 轨迹序列, dataframe格式. 必须包含经纬度坐标.
        - thresholdEI: 混乱指数阈值, 高于即判断为满足约束.
        - groups: 区间分成多少组. 组数越大, 判断为不同组的可能性越大.

    Return:
        返回布尔值, True为df序列通过约束验证.
    """

    # FIXME 单点情况的处理问题. activity的情况较少
    assert df.shape[0] > 1, 'too short pieces of gps'

    theats = list(t for t in Df2T(df))
    ei = entropyIndex(theats, groups)
    if ei > thresholdEI:
        return True


class DPCLink_TE(DPCLink):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.logger = logging.getLogger('dpc.DPCL_TE')

    # TODO 暂用，注入数据。
    def getLabel(self, df_label):
        self.dfl = df_label

    def _ei(self, sq: list, d: int) -> float:
        """计算一个序列的混乱指数, d为分组数"""

        # 生成分组区间
        ndNs = {}
        for l, r in Two(np.linspace(0, 360, d + 1)):
            ndNs[str(l) + '-' + str(r)] = []
            ndNs[str(l) + '-' + str(r)].append(pd.Interval(left=l, right=r, closed='right'))
            ndNs[str(l) + '-' + str(r)].append(0)

        # 由于不是循环区间，0无法被认为落到(:360]的区间. 所以要把0转换成360
        sq = [s if s else 360 for s in sq]

        # 计算落入区间内数量
        for s in sq:
            for n in ndNs:
                if s in ndNs[n][0]:
                    ndNs[n][1] = ndNs[n][1] + 1
        ndNs = [ndNs[n][1] for n in ndNs]
        ndNs = [x / sum(ndNs) for x in ndNs]

        ei = -sum([x * math.log(x, math.e) for x in ndNs if x != 0])
        return ei

    def TE(self, thresholdEI: float = 0.5, groups: int = 10):
        """Temporal sequence constraint and Entropy constraint"""

        def _eiCon(start: int, end: int):
            """判断轨迹序列是否满足混乱指数约束, 满足则将该段轨迹赋值活动点"""
            start = start
            end = end
            pass
            # FIXME 单点情况的处理问题. activity的情况较少
            if start == end:
                # if self.df.loc[start, 'rho'] > 1.5:
                # self.df.loc[start, 'type'] = 'activity'
                return

            # FIXME 序列混有驻足点和其它点，混乱指数的递增性
            theats = self.df.loc[start:end]
            theats = list(t for t in Df2T(self.df.loc[start:end]))
            ei = self._ei(theats, groups)
            if ei > thresholdEI:
                self.df.loc[start:end, 'type'] = 'activity'
            # self.logger.debug('cutting: ' + str(self.df.loc[start:end].index.tolist()[0]) + '-' + str(self.df.loc[start:end].index.tolist()[-1]) +
            #                   '. len: ' + str(int(end - start + 1)) +
            #                   '. ei: ' + str(ei))

        # 新增或清空type类型
        self.df['type'] = ''

        # 清空ei值
        # self.df['ei'] = -1

        tmp = self.df[self.df.clusterID != -1]
        for clusterID, dfc in tmp.groupby('clusterID'):
            start = int(dfc.index[0])
            # self.logger.info('Cutting and processing clusters ' + str(int(clusterID)))

            for (index_pre, index_nex), (track_pre, track_nex) in FourDf(dfc):
                if index_nex == None:
                    _eiCon(start, index_pre)
                elif index_nex == index_pre + 1 and track_pre == track_nex:
                    continue
                else:
                    _eiCon(start, index_pre)
                    start = index_nex

        _eiCon(start, index_pre)

        return self.df

    def tempory(self):
        """对dpc的df进行序列切片, 并返回切片"""
        pieces = tempory(self.df)
        self.df['piecesID'] = -1

        for i, pie in enumerate(pieces):
            indexs = self.df[pie[0]:pie[1]].index.tolist()
            self.df.loc[indexs, 'piecesID'] = i+1
        return pieces

    def entropy(self, pieces: list, thresholdEI: float = 0.5, groups: int = 8, minPots: int = 2, minRho: float = 0.1) -> pd.DataFrame:
        """"对切片后的序列进行混乱指数判断, 并返回df

        Args:
            - pieces: 切片序列, list of list
            - thresholdEI: 混乱指数阈值
            - groups: 计算混乱指数时的分组数
            - minPots: 一条提取的驻足点应满足的最小数值

        Return:
            - 与输入的df保持一致, 但是多了一列type, 值为activity的是驻足点.
        """

        self.df['type'] = ''
        for p in pieces:
            if p[1]-p[0]+1 < minPots:
                continue
            df_tmp = self.df.loc[p[0]:p[1]]
            # df_tmp = df_tmp[df_tmp.rho > minRho]
            if eiPicecs(df_tmp, thresholdEI, groups):
                self.df.loc[p[0]:p[1], 'type'] = 'activity'

        return self.df

    def cluster(self, minRho: float = 0.0):
        logger = logging.getLogger('dpc.DPCLTE.cluster')

        def group(i, id):
            self.df.loc[i, 'clusterID'] = id
            # logger.debug(str(i) + ' was classified as cluster ' + str(id))

            for j in self.df[self.df.toh == i].index.tolist():
                group(j, id)

        # 聚类中心不应该指向比它密度更高的聚类中心
        self.df.loc[self.centers, 'toh'] = -1

        # 聚类前清空原聚类结果, 全部复位噪声
        self.df['clusterID'] = -1

        for idCluster, indexCenter in enumerate(self.centers):
            group(indexCenter, idCluster + 1)
            # logger.info(str(self.df[self.df.clusterID == idCluster + 1].shape[0]) + 'points in cluster ' + str(idCluster + 1) + ' with center point ' + str(indexCenter))

        # 低密度点clusterID设为-1, 认为是噪声.
        self.df.loc[self.df['rho'] < minRho, 'clusterID'] = -1

        return self.df
