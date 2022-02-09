import numpy as np
import pandas as pd


class Load:
    def __init__(self) -> None:
        pass

    def readLine(file_path):
        x = []
        y = []
        with open(file_path) as f:
            for l in f.readlines():
                line = l.split()
                x.append(float(line[0]))
                y.append(float(line[1]))

        x = np.array(x)
        y = np.array(y)

        return x, y

    def readPd(self, file_path):
        data = pd.read_csv(file_path, delim_whitespace=True, names=['x', 'y', 'none'])[['x', 'y']]

        return data

    def disM_line(self, file_path):
        data = pd.read_csv(file_path, names=['x', 'y', 'd'], sep=' ')[['x', 'y']]
        N = max(data.max())
        disMatrix = np.zeros((N, N), dtype=np.float64)
        with open(file_path) as f:
            for l in f.readlines():
                line = l.split()
                i = int(line[0])-1
                j = int(line[1])-1
                disMatrix[i, j] = float(line[2])
                disMatrix[j, i] = float(line[2])
        return disMatrix
