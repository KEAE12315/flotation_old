import matplotlib.image as mpimg
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
data=pd.read_csv('dataset/example_distances.txt',names=['x','y','d'],sep=' ')
print(max(data[['x','y']].max()))
ml=pd.MultiIndex.from_frame(data[['x','y']])
data.drop(columns=['x','y'],inplace=True)
data=data.set_index(ml)
data=data.unstack()
# data=data.reset_index()
print((data))