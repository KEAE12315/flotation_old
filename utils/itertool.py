import pandas as pf
from distance import calc_azimuth


class Two:

    def __init__(self, it):
        self.it = it
        self.i = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.it)-2:
            self.i = self.i+1
            return self.it[self.i], self.it[self.i+1]
        else:
            raise StopIteration


class FourDf():
    """返回df的index和trackID, 各自一前一后的数"""

    def __init__(self, df):
        self.df = df
        self.i = -1
        self.bd = self.df.shape[0]-2

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.bd:
            self.i = self.i+1
            return (self.df.iloc[self.i].name, self.df.iloc[self.i+1].name), (self.df.iloc[self.i].trackID, self.df.iloc[self.i+1].trackID)
        else:
            raise StopIteration


class Df2T:
    """返回df相邻两轨迹点的方位角"""

    def __init__(self, df):
        self.df = df
        self.i = -1
        self.bd = self.df.shape[0]-2

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.bd:
            self.i = self.i+1
            return calc_azimuth(self.df.iloc[self.i].lat,
                                self.df.iloc[self.i].lng,
                                self.df.iloc[self.i+1].lat,
                                self.df.iloc[self.i+1].lng)
        else:
            raise StopIteration


if __name__ == '__main__':
    for i in Two(list(range(5))):
        print(i)
