import torch
from torch.utils.data import Dataset
import pandas as pd


class FeedDataset(Dataset):
    def __init__(self, df, ls=10):
        self.df = df.reset_index(drop=True)
        self.ls = ls

        self.cols_x = ['% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density']
        self.cols_y = ['% Iron Concentrate', '% Silica Concentrate']

    def __getitem__(self, index):
        x = self.df.loc[index:index+self.ls, self.cols_x]
        x = x.values.flatten()
        y = self.df.loc[index+self.ls, self.cols_y]

        x = torch.tensor(x.tolist(), dtype=torch.float32)
        y = torch.tensor(y.tolist(), dtype=torch.float32)

        return x, y

    def __len__(self):
        return self.df.shape[0]-self.ls

    def size(self):
        x_n = (self.ls+1)*self.cols_x.__len__()
        y_n = self.cols_y.__len__()
        return x_n, y_n


class Normalize():
    def __init__(self, norm_path='model/norm.csv') -> None:
        self.cols = ['Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density', '% Iron Concentrate', '% Silica Concentrate']
        self.norm_path = norm_path

    def inite(self, df):
        inter = df[self.cols].max()-df[self.cols].min()
        miner = df[self.cols].min()

        norm = pd.concat([inter, miner], axis=1)
        norm.columns = ['inter', 'miner']
        norm = norm.T
        norm.to_csv(self.norm_path)

    def code(self, df):
        norm = pd.read_csv(self.norm_path, index_col=0)

        df[['% Iron Feed', '% Silica Feed']] = df[['% Iron Feed', '% Silica Feed']]/100
        for c in self.cols:
            df[c] = (df[c]-norm.loc['miner', c])/norm.loc['inter', c]

        return df

    def decode(self, df):
        norm = pd.read_csv(self.norm_path, index_col=0)

        # df[['% Iron Feed', '% Silica Feed']] = df[['% Iron Feed', '% Silica Feed']]*100
        # for c in self.cols:
        #     df[c] = df[c]*norm.loc['inter', c] + norm.loc['miner', c]

        df[0] = df[0]*norm.loc['inter', '% Iron Concentrate']+norm.loc['miner', '% Iron Concentrate']
        df[1] = df[1]*norm.loc['inter', '% Silica Concentrate']+norm.loc['miner', '% Silica Concentrate']

        return df


def create_data(df, split=0.8, ls=10):
    df = df.drop(columns=['X1', 'date'])

    norm = Normalize()
    norm.inite(df)
    df = norm.code(df)

    s = int(len(df)*split)
    train = FeedDataset(df.loc[:s], ls)
    test = FeedDataset(df.loc[s:], ls)
    x_n, y_n = train.size()

    return train, test, x_n, y_n
