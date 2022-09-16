## dpc.df.columns
- rawIndex: 在所有数据中的原索引
- lat: 维度
- lng: 经度
- datetime: 年月日时分秒
- trackID: 第几条轨迹
- rho: 局部密度
- delta: 高密度距离
- toh: 最近更高密度点
- gamma: 综合判据
- clusterID: 所属簇编号
- piecesID: 切片序号, 但每一次切割都会重新编号, 不在簇内的为-1
- type: 点类型

## DPCLink

将距离函数改为距离一致性函数. 但距离是越小越好, 这个是越大越好, 计算局部密度时需要调整为大于号. 此时dc的含义为指数的阈值. 由于指数设置对近距离敏感, 导致按比例选择dc时对参数p敏感.

FourDF边界问题

```
total points: 9080
tp: 1987
tn: 1069
fp: 5843
fn: 181

accuracy: 33.66%
precision: 25.38%
recall: 91.65%
F index: 39.75%
```

## dataset descrption

### Geolife Trajectories 1.3

### GTMarker

label data of Geolife Trajectories 1.3

# FIXME
rho 归一化
ei boxplot 按种类划分