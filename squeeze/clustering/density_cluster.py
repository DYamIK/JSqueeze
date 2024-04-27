from typing import List
import seaborn as sns
import numpy as np
from loguru import logger
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from Jsqueeze.clustering.cluster import Cluster
from Jsqueeze.squeeze_option import SqueezeOption
from kneed import KneeLocator
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def smooth(arr, window_size):
    new_arr = np.convolve(arr, np.ones(window_size), mode="valid") / window_size
    new_arr = np.concatenate([arr[:window_size - 1], new_arr])
    assert np.shape(new_arr) == np.shape(arr)
    return new_arr


class DensityBased1dCluster(Cluster):
    def __init__(self, option: SqueezeOption):
        super().__init__(option)
        assert option.density_estimation_method in {'kde', 'histogram'}
        self.density_estimation_func = {
            "kde": self._kde,
            "histogram": self._histogram,
        }[option.density_estimation_method]

    def _kde(self, array: np.ndarray):
        kernel = gaussian_kde(array, bw_method=self.option.kde_bw_method, weights=self.option.kde_weights)
        samples = np.arange(np.min(array), np.max(array), 0.01)
        kde_sample = kernel(points=samples)
        conv_kernel = self.option.density_smooth_conv_kernel
        kde_sample_smoothed = np.convolve(kde_sample, conv_kernel, 'full') / np.sum(conv_kernel)
        return kde_sample_smoothed, samples

    def _histogram(self, array: np.ndarray):
        def _get_hist(_width):
            if _width == 'auto':
                _edges = np.histogram_bin_edges(array, 'auto').tolist()
                _edges = [_edges[0] - 0.1 * i for i in range(-5, 0, -1)] + _edges + [_edges[-1] + 0.1 * i for i in
                                                                                     range(1, 6)]
            else:
                _edges = np.arange(array_range[0] - _width * 6, array_range[1] + _width * 5, _width)
            h, edges = np.histogram(array, bins=_edges, density=True)
            h /= 100.
            # conv_kernel = self.option.density_smooth_conv_kernel
            # h = np.convolve(h, conv_kernel, 'full') / np.sum(conv_kernel)
            return h, np.convolve(edges, [1, 1], 'valid') / 2
        array_range = np.min(array), np.max(array)
        width = self.option.histogram_bar_width

        return _get_hist(width)

    def _cluster(self, array, density_array: np.ndarray, bins, plot=False):
        order = 1
        extreme_max_indices = argrelextrema(
            density_array, comparator=lambda x, y: x > y,
            axis=0, order=order, mode='wrap')[0]
        extreme_min_indices = argrelextrema(
            density_array, comparator=lambda x, y: x <= y,
            axis=0, order=order, mode='wrap')[0]
        extreme_max_indices = list(filter(lambda x: density_array[x] > 0, extreme_max_indices))
        if plot:
            for idx in extreme_max_indices:
                plt.axvline(bins[idx], linestyle="-", color="red", label="relmax", alpha=0.5, linewidth=0.8)
            for idx in extreme_min_indices:
                plt.axvline(bins[idx], linestyle="--", color="blue", label="relmin", alpha=0.5, linewidth=0.8)

        cluster_list = []
        boundaries = np.asarray([float('-inf')] + [bins[index] for index in extreme_min_indices] + [float('+inf')])
        if self.option.max_normal_deviation == 'auto':
            mu = np.mean(np.abs(array))
            max_normal = mu
            logger.debug(f"max normal {max_normal}")
            self.option.max_normal_deviation = max_normal
        for index in extreme_max_indices:
            left_boundary = boundaries[np.searchsorted(boundaries, bins[index], side='right') - 1]
            right_boundary = boundaries[np.searchsorted(boundaries, bins[index], side='left')]
            cluster_indices = np.where(
                np.logical_and(
                    array <= right_boundary,
                    array >= left_boundary,
                    )
            )[0]
            cluster = array[cluster_indices]
            mu = np.mean(np.abs(cluster))
            logger.debug(f"({left_boundary, right_boundary}, {mu})")
            if np.abs(mu) < self.option.max_normal_deviation or len(cluster) <= 0:
                continue
            cluster_list.append(cluster_indices)
        return cluster_list

    def compute_core_distances(self, data, minPts):

        nbrs = NearestNeighbors(n_neighbors=minPts).fit(data)
        
        distances, _ = nbrs.kneighbors(data)
        
        core_distances = distances[:, -1]
    
        return core_distances
    
    def compute_reachability_distances(data, minPts):
        nbrs = NearestNeighbors(n_neighbors=minPts).fit(data)
        distances, _ = nbrs.kneighbors(data)
        core_distances = distances[:, -1]
        n_samples = data.shape[0]
        reachability_distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                reachability_distances[i, j] = max(core_distances[i], abs(data[i] - data[j]))
    
        return reachability_distances

    def __call__(self, array):
        array = array.copy()

        # array=array.reshape(-1,1)
        # array=self.compute_core_distances(array, 1)

        density_array, bins = self.density_estimation_func(array)
        # normal_idxes = self._find_normal_indices(array, density_array, bins)
        # density_array, bins = self.density_estimation_func(array[~normal_idxes])
        density_array = np.copy(density_array)
        if self.option.cluster_smooth_window_size == "auto":
            # window_size = max(int(np.log(np.count_nonzero(bins[density_array > 0.])) / np.log(10)), 1)
            window_size = max(np.count_nonzero(density_array > 0) // 10, 1)
        else:
            window_size = self.option.cluster_smooth_window_size
        smoothed_density_array = smooth(density_array, window_size)
        if self.option.debug:
            fig, ax1 = plt.subplots(figsize=(3.6, 1.8))
            sns.distplot(array, bins='auto', label="density", hist=True, kde=False, norm_hist=True, ax=ax1)
            ax1.set_ylim([0, None])
            # ax2 = ax1.twinx()
            # ax2.plot(bins, smoothed_density_array, label="smoothed", linestyle="-.")
            # ax2.set_ylim([0, None])
        clusters = self._cluster(array, smoothed_density_array, bins, plot=self.option.debug)
        if self.option.debug:
            for cluster in clusters:
                left_boundary, right_boundary = np.min(array[cluster]), np.max(array[cluster])
                # plt.axvline(left_boundary, c='C0', alpha=0.5, linestyle='--')
                # plt.axvline(right_boundary, c='C1', alpha=0.5, linestyle=':')
                logger.debug(f"cluster: [{left_boundary}, {right_boundary}]")
            by_label1 = dict(zip(*reversed(ax1.get_legend_handles_labels())))
            # by_label2 = dict(zip(*reversed(ax2.get_legend_handles_labels())))
            by_label2 = {}
            # logger.debug(f"{by_label1}, {by_label2}")
            plt.legend(
                list(by_label1.values()) + list(by_label2.values()),
                list(by_label1.keys()) + list(by_label2.keys()), bbox_to_anchor=(0.47, 0.5)
            )
            plt.xlim([-0.9, 1])
            # plt.title(self.option.density_estimation_method)
            plt.xlabel('deviation score')
            plt.ylabel('pdf')
            plt.tight_layout()
            # plt.show()
            plt.savefig(self.option.fig_save_path.format(suffix="_density_cluster"))
            plt.close()
        return clusters

class HDBSCAN1dCluster(Cluster):
    def __init__(self, option: SqueezeOption,min_simples=5):
        super().__init__(option)
        #assert option.density_estimation_method in {'kde', 'histogram'}
        # self.density_estimation_func = {
        #     "kde": self._kde,
        #     "histogram": self._histogram,
        # }[option.density_estimation_method]
        self.min_simples=min_simples
        self.dbscan=hdbscan.HDBSCAN(min_samples=min_simples)
        
    def __call__(self, array:np.ndarray) -> List[np.ndarray]:
        array=array.copy()
        clusters = self.dbscan.fit_predict(array.reshape(-1,1))
        cluster_list=[]
        for i in range(np.min(clusters),np.max(clusters)):
            if np.mean(np.abs(array[np.where(clusters==i)[0]]))>=self.option.max_normal_deviation:
                cluster_list.append(np.where(clusters==i))
        return cluster_list
    
class DBSCAN1dCluster(Cluster):
    def __init__(self, option: SqueezeOption,eps=0.01,min_simples=5):
        super().__init__(option)
        #assert option.density_estimation_method in {'kde', 'histogram'}
        # self.density_estimation_func = {
        #     "kde": self._kde,
        #     "histogram": self._histogram,
        # }[option.density_estimation_method]
        self.min_simples=min_simples
        self.eps=eps
        self.dbscan=DBSCAN(eps,min_samples=min_simples)
        
    def __call__(self, array:np.ndarray) -> List[np.ndarray]:
        array=array.copy()
        clusters = self.dbscan.fit_predict(array.reshape(-1,1))
        cluster_list=[]
        for i in range(np.min(clusters),np.max(clusters)):
            if np.mean(np.abs(array[np.where(clusters==i)[0]]))>=self.option.max_normal_deviation:
                cluster_list.append(np.where(clusters==i))
        return cluster_list