from utils.itertool import Two
import math
import numpy as np
import pandas as pd


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
