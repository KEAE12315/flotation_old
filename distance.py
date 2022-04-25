from haversine import haversine
import math


def norm2(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def coh(p, q, delta=0.1):
    """计算区域一致性指数. 值越大, 两坐标点越有可能属于同一区域.

    Args:
        p : 上一个坐标点, 保证能用点运算符找到lat,lng,datetime属性. 其中datetime属性为datetime类型数据.
        q : 下一个坐标点, 保证能用点运算符找到lat,lng,datetime属性. 其中datetime属性为datetime类型数据.
        delta (float, optional): 放缩参数, 将距离和速度放缩到同一量级. Defaults to 0.1.

    Raises:
        ValueError: 坐标点未按时间顺序排列时, 抛出此异常.

    Returns:
        float: 区域一致性指数.
    """

    # 距离km，时间h，速度km/h
    distance = haversine([p.lat, p.lng], [q.lat, q.lng])
    duration = (q.datetime - p.datetime).total_seconds() / 60 / 60
    if duration < 0:
        print('数据未按照时间顺序排列, 请检查')
        raise ValueError

    speed = distance / duration

    a = distance / delta
    b = speed
    c = math.atan(a + b)
    return c
