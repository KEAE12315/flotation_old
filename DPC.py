from distance import norm2
import pandas as pd
import numpy as np


class DPC:
    def __init__(self, data) -> None:
        self.df = data
        self.N = self.df.shape[0]

    def loadDis(self, dis):
        self.dis = dis

    def calDc_raw(self, p):
        tmp = np.tril(self.dis)
        tmp = tmp.ravel()
        tmp = tmp[np.where(tmp)]
        tmp = np.sort(tmp)
        self.dc = tmp[round(len(tmp)*p)]
        print('dc: '+str(self.dc))

    def calDis(self):
        disMatrix = np.zeros((self.N, self.N))
        for i, ix, iy in self.df[['x', 'y']].itertuples():
            for j, jx, jy in self.df.loc[i+1:, ['x', 'y']].itertuples():
                d = norm2(ix, iy, jx, jy)
                disMatrix[i, j] = d
                disMatrix[j, i] = d

        self.dis = disMatrix

    def calRho(self, kernel='gaussian'):
        rho = np.zeros(self.N)

        if kernel == 'cutoff':
            for i, *_ in self.df[:self.N-2].itertuples():
                for j, *_ in self.df[i+1:self.N-1].itertuples():
                    if self.dis[i, j] < self.dc:
                        rho[i] = rho[i]+1
                        rho[j] = rho[j]+1

        elif kernel == 'gaussian':
            for i, *_ in self.df[:self.N-2].itertuples():
                for j, *_ in self.df[i+1:self.N-1].itertuples():
                    tmp = np.exp(-(self.dis[i, j]/self.dc)**2)
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

    def getCen(self, n):
        self.centers = self.df.sort_values(by='gamma', ascending=False).index.values[:n]
        print(self.centers)

    def cluster(self):
        def group(i, id):
            if id == 4:
                print(i)
            self.df.loc[i, 'clusterID'] = id
            for j in self.df[self.df.toh == i].index.tolist():
                group(j, id)

        self.df.loc[self.centers, 'toh'] = -1
        for i, c in enumerate(self.centers):
            print(i, c)
            group(c, i+1)


if __name__ == "__main__":
    from load import *

    load = load_xy('dataset/xy/')
    data = next(load)
    data = next(load)
    dpc = DPC(data[['x', 'y']])
    dpc.calDis()
    dpc.calDc_raw(0.01)
    dpc.calRho()
    dpc.calDel()
    dpc.calGam()

    dpc.getCen(15)

    import matplotlib.pyplot as plt
    dpc.cluster()
    print(np.sort(dpc.df['clusterID'].unique()))

    fig, ax = plt.subplots()
    for i, ri in dpc.df.drop(dpc.centers).iterrows():
        ax.arrow(ri['x'], ri['y'], dpc.df.loc[ri['toh'], 'x']-ri['x'], dpc.df.loc[ri['toh'], 'y']-ri['y'], lw=0.01, head_width=0.02)
    # dpc.df.plot.scatter(x="x", y='y', s=1, c='rho', cmap='viridis', ax=ax)

    dpc.df.plot.scatter(x='x', y='y', c='clusterID', cmap='tab20', ax=ax)
    ax.set_axis_off()
    plt.show()
