from functools import total_ordering
import numpy as np
import operator


@total_ordering
class Cluster(object):
    def __init__(self, cl_size: int, clusters: list = None, nodes: list = None):
        self.nodes = set()

        self.children = []
        self.falling_out_points = []

        assert clusters is not None or nodes is not None
        if clusters is not None:
            for cluster in clusters:
                self.nodes.update(cluster.nodes)
                self.children.append(cluster)
        else:
            self.nodes.update(nodes)
        self.frozennodes = frozenset(self.nodes)
        self.__hash = hash(self.frozennodes)

    def append(self, weight: float, clusters: list):
        for cluster in clusters:
            self.nodes.update(cluster.nodes)
        self.frozennodes = frozenset(self.nodes)
        self.__hash = hash(self.frozennodes)
        return self
    
    def __iter__(self):
        for child in self.children:
            yield child
    
    def __contains__(self, node):
        return node in self.nodes
    
    def __len__(self):
        return len(self.nodes)
    
    def __hash__(self):
        return self.__hash
    
    def __eq__(self, other):
        return self.__hash == other.__hash

    def __lt__(self, other):
        return self.__hash < other.__hash


def simple_clusterer(rows, cols, weights, n, max_cl=5, fraction=80):
    edges = np.c_[rows, cols, weights]
    edges = edges[edges[:, -1].argsort()]

    clusters = {}
    for node_id in range(n):
        clusters[node_id] = Cluster(cl_size=1, nodes=[node_id])

    for i, j, weight in edges:
        if (len(clusters[i]) >= max_cl) or (len(clusters[j]) >= max_cl):
            continue
            
        if clusters[i] is clusters[j]:
            continue

        cluster = Cluster(cl_size=1, clusters=[clusters[i], clusters[j]])
        clusters.update({l: cluster for l in cluster.nodes})
        
    labels_ = np.full(fill_value=-1, shape=len(clusters))
    c = 0
    for cluster in list(set(clusters.values())):
        labels_[np.array(list(cluster.nodes)).astype(int)] = c
        c += 1
    return labels_

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def clusterise_data(X, Y, M, min_samples=1, min_cluster_size=2, verbose=True, max_cl=3, fraction=80):
    """
    
    """
    distance_matrix = csr_matrix(squareform(pdist(X[:, :3])))
    weights = distance_matrix.data
    rows, cols = distance_matrix.nonzero()

    max_weight = np.percentile(weights, q=fraction)
    mask = weights <= max_weight
    weights, rows, cols = weights[mask], rows[mask], cols[mask]

    # possible variant wth minimum spanning tree
    # mst = minimum_spanning_tree(distance_matrix)
    # weights = mst.data
    # rows, cols = mst.nonzero()
    
    labels_ = simple_clusterer(rows=rows, cols=cols, weights=weights, n=len(X), max_cl=max_cl, fraction=fraction)
            
    X_cluster = []
    Y_cluster = []
    M_cluster = []
    for label in np.unique(labels_):
        if label == -1:
            if (labels_ == label).sum() == 1:
                continue
            for row, y in zip(X[labels_ == label], Y[labels_ == label]):
                X_cluster.append(row.reshape(1, -1))
                Y_cluster.append(np.array([y]))
            continue

        X_cluster.append(X[labels_ == label])
        Y_cluster.append(Y[labels_ == label])
        M_cluster.append(M[labels_ == label])
    if verbose:
        print('Compression rate:', len(Y) / len(Y_cluster))
        number_mixed = 0
        for y in Y_cluster:
            values, counts = np.unique(y, return_counts=True)
            counts_normed = counts / counts.sum()
            if counts_normed.max() < 0.9:
                number_mixed += counts.sum()
        print('Mixed fraction:', 100 * number_mixed / len(Y))
        print()
        
    return X_cluster, Y_cluster, M_cluster