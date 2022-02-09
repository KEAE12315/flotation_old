from distance import norm2
import pandas as pd
import numpy as np


class DPC:
    def __init__(self, data) -> None:
        self.df = data
        self.N = self.df.shape[0]

    def loadDis(self, dis):
        self.dis = dis

    def calDis(self):
        disMatrix = np.zeros((self.N, self.N))
        for i, ix, iy in self.df[['x', 'y']].itertuples():
            for j, jx, jy in self.df.loc[i+1:, ['x', 'y']].itertuples():
                d = norm2(ix, iy, jx, jy)
                disMatrix[i, j] = d
                disMatrix[j, i] = d

        self.dis = disMatrix

    def calRho(self, dc=1, kernel='cutoff'):
        rho = np.zeros(self.N)

        if kernel == 'cutoff':
            for i, *_ in self.df[:self.N-2].itertuples():
                for j, *_ in self.df[i+1:self.N-1].itertuples():
                    if self.dis[i, j] < dc:
                        rho[i] = rho[i]+1
                        rho[j] = rho[j]+1

        elif kernel == 'gaussian':
            for i, *_ in self.df[:self.N-2].itertuples():
                for j, *_ in self.df[i+1:self.N-1].itertuples():
                    tmp = np.exp(-(self.dis[i, j]/dc)**2)
                    rho[i] = rho[i]+tmp
                    rho[j] = rho[j]+tmp

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
                print(i)
                delta[i] = max(self.dis[i, :])
                toh[i] = -1

        self.df['delta'] = delta
        self.df['toh'] = toh

    def calGam(self):
        self.df['gamma'] = self.df['rho']*self.df['delta']

        self.centers = self.df.sort_values(by='gamma', ascending=False).index.values[:2]
        print(self.centers)

    def cluster(self):
        def group(i, id):
            self.df.loc[i, 'clusterID'] = id
            for j in self.df[self.df.toh == i].index.tolist():
                group(j, id)

        for i, c in enumerate(self.centers):
            group(c, i+1)


if __name__ == "__main__":
    from load import Load
    load = Load()
    data = load.readPd('dataset/gauss_data.txt')
    cluster = DPC(data)
    cluster.calRho(dc=2)
    cluster.calDel()
    print(cluster.data)
