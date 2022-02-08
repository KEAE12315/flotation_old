import matplotlib.image as mpimg
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
plt.figure(figsize=(12, 12), dpi=300)


plt.ion()   # matplotlib interactivate mode
for i, ri in data.iterrows():
    plt.scatter(data['x'].values.tolist(), data['y'].values.tolist(), s=1, c=data['rho'].values)
    plt.scatter(ri['x'], ri['y'], c='red', label=str(i))
    plt.legend()

    plt.pause(0.1)  # pause 0.1 second
    # input()
    plt.clf()
    plt.show()
