from haversine import haversine
import math

def norm2(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def coh(self, p, q) -> bool:
    """计算两个坐标点之间的区域一致性指数.
    """
    # 距离km，时间h，速度km/h
    distance = haversine([p.lat, p.lng], [q.lat, q.lng])
    duration = abs((p.days - q.days) * 24)
    speed = distance / duration

    a = distance / self.delta
    b = speed
    c = math.exp(-a - b)
    return c
