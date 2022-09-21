from utils.itertool import Two
import math
import numpy as np
import pandas as pd


def theil(x: list):
    """theil index"""
    x = np.array(x)
    mean = np.mean(x)
    x = list(map(lambda x: x/mean*np.log(x/mean), x))
    x = np.array(x)
    x = np.sum(x)/x.size
    return x


def theilL(x: list):
    """theil index"""
    x = np.array(x)
    mean = np.mean(x)
    x = list(map(lambda x: np.log(mean/x), x))
    x = np.array(x)
    x = np.sum(x)/x.size
    return x


def entroyIndex(sq: list, d: int) -> float:
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


def bonferroni(x):
    """计算离散邦费罗尼指数"""
    xmin = min(x)
    xmax = max(x)
    if xmax-xmin != 0:
        for i, v in enumerate(x):
            x[i] = (v-xmin) / (xmax-xmin)
        x.sort()
    b = 0
    N = len(x)
    tmp = sum(x)
    for i in range(len(x)):
        p = (i + 1) / N
        q = sum(x[:i + 1]) / tmp
        b = b + (1 - q / p)

    b = b / (N - 1)
    return b


def centerIndex(df, d):
    df['wm'] = -1
    for p in df[1:-2].itertuples():
        a = df.loc[:p.Index, 'gamma'].sum() / (p.Index+1)-df.loc[p.Index+1, 'gamma']

        sum = 0
        for i in range(p.Index+1):
            sum = sum+d[df.loc[i, 'rawIndex']-1, p.rawIndex-1]
        b = sum/(p.Index+1)
        df.loc[p.Index, 'wm'] = a
    return df
