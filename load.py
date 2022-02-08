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
        data = pd.read_csv(file_path, delim_whitespace=True,names=['x', 'y', 'none'])[['x','y']]

        return data
