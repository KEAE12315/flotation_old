import pandas as pd
from utils.distance import calc_azimuth
from abc import abstractmethod, ABCMeta


class Two:

    def __init__(self, it):
        self.it = it
        self.i = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.it) - 2:
            self.i = self.i + 1
            return self.it[self.i], self.it[self.i + 1]
        else:
            raise StopIteration


class BefAftN(metaclass=ABCMeta):
    """ 返回前后N个数 """

    def __init__(self, it, afterN: int, beforeN: int = None, startIndex: int = 0):
        self.it = it
        self.aftN = afterN

        if beforeN is None:
            beforeN = afterN
        self.befN = beforeN

        if startIndex < 0 or startIndex >= len(self.it):
            raise ValueError('start index out of range')
        self.i = startIndex - 1

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass


class EIPieces(BefAftN):
    """ 定长滑动窗口 """

    def __init__(self, indexsDf, afterN: int, beforeN: int = None, startIndex: int = 0):
        super().__init__(indexsDf, afterN, beforeN, startIndex)

    def __next__(self):
        self.i = self.i + 1

        if self.i >= len(self.it):
            raise StopIteration

        beg = self.i - self.befN
        end = self.i + self.aftN
        offset = 0

        if beg < 0:
            beg = 0
            # offset = self.befN - self.i
            # if end + offset > len(self.it) - 1:
            #     end = len(self.it) - 1
            # else:
            #     end = end + offset

        if end > len(self.it) - 1:
            offset = end - len(self.it) + 1
            end = len(self.it) - 1
            # if beg - offset < 0:
            #     beg = 0
            # else:
            #     beg = beg - offset

        return self.it[beg:end + 1], self.it[self.i]


class FourDf():
    """返回df的index和trackID, 各自一前一后的数"""

    def __init__(self, df):
        self.df = df
        self.i = -1
        self.bd = self.df.shape[0] - 2

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.bd:
            self.i = self.i + 1
            return (self.df.iloc[self.i].name, self.df.iloc[self.i + 1].name), (self.df.iloc[self.i].trackID, self.df.iloc[self.i + 1].trackID)
        elif self.i == self.bd:
            self.i = self.i + 1
            return (self.df.iloc[self.i].name, None), (self.df.iloc[self.i].trackID, None)
        else:
            raise StopIteration


class Df2T():
    """返回df相邻两轨迹点的方位角"""

    def __init__(self, df):
        self.df = df
        self.i = -1
        self.bd = self.df.shape[0] - 2

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.bd:
            self.i = self.i + 1
            return calc_azimuth(self.df.iloc[self.i].lat,
                                self.df.iloc[self.i].lng,
                                self.df.iloc[self.i + 1].lat,
                                self.df.iloc[self.i + 1].lng)
        else:
            raise StopIteration


if __name__ == '__main__':
    for i in EIPieces(list(range(20)), 3):
        print(i)
