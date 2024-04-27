from .cluster import *
from .density_cluster import *


def cluster_factory(option: SqueezeOption):
    method_map = {
        "density": DensityBased1dCluster,
        "DBSCAN": DBSCAN1dCluster,
        "HDBSCAN": HDBSCAN1dCluster,
    }
    # 在这里，cluster_method 是一个字符串，指定要使用的一维聚类方法的名称。当前实现中只提供了一种方法，即基于密度的一维聚类（DensityBased1dCluster）。
    return method_map[option.cluster_method](option)
