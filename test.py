# %%
# 读取轨迹数据
from utils.load import *
import pandas as pd

# 读取标记数据
df_labels = pd.read_json("/home/plw/dataset/GTMarker/001/g2.json")

# 读取原始轨迹
dataset_path = '/home/plw/dataset/Geolife Trajectories 1.3/Data/001/Trajectory/'
dataset = enumerate(LoadG(dataset_path, 2))
dfs = pd.DataFrame()
for i, df in dataset:
    df['trackID'] = i + 1
    dfs = pd.concat([dfs, df], ignore_index=True)
dfs.reset_index(drop=True)
pass

# %%
print(list(range(1, 10, 2)))
# %%
