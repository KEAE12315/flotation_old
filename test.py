from utils.load import LoadG

# 读取文件为迭代器, 可指定读取数量
dataset_path = 'dataset/Geolife Trajectories 1.3/Data/001/'
dataset = enumerate(LoadG(dataset_path))
# dataset = enumerate(itertools.islice(LoadG(dataset_path), 0, None))

for i in dataset:
    print(i)

