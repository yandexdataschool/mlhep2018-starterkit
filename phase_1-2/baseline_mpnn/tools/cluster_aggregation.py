import numpy as np
import hdmedians as hd


def principal_axes(X):
    N = len(X)
    inertia = (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))
    e_values, e_vectors = np.linalg.eig(inertia)

    order = np.argsort(e_values)
    eval3, eval2, eval1 = e_values[order] / N
    axis3, axis2, axis1 = e_vectors[:, order].transpose()

    return (eval1, eval2, eval3), (axis1, axis2, axis3)


def aggregate_clusters(X_cluster, Y_cluster, M_cluster):
    """
    Calculate aggregated statistics for clusterised data
    """
    X_condensed = []
    Y_condensed = []
    clusters = []
    for x, y, m in zip(X_cluster, Y_cluster, M_cluster):
        x_condensed = np.array(hd.geomedian(x[:, :3], axis=0))        
        X_condensed.append(x_condensed)
        
        counts = y.sum(axis=0)
        y_condensed = len(counts) * [0]
        y_condensed[np.argmax(counts)] = 1
        Y_condensed.append(y_condensed)
        
        (eval1, eval2, eval3), (axis1, axis2, axis3) = principal_axes(x[:, :3])
        cluster = {
            'X': x,
            'Y': y,
            'M': m,
            'X_condensed': x_condensed,
            'Y_condensed': y_condensed,
            'axis1': axis1,
            'axis2': axis2,
            'axis3': axis3,
            'eval1': eval1,
            'eval2': eval2,
            'eval3': eval3,
            'E_mean': np.mean(x[:, 3]),
            'E_median': np.median(x[:, 3]),
            'E_std': np.std(x[:, 3])
        }
        
        clusters.append(cluster)
    X_condensed = np.array(X_condensed)
    Y_condensed = np.array(Y_condensed)
    
    data = {
        'X_condensed': X_condensed,
        'Y_condensed': Y_condensed,
        'clusters': clusters
    }
    
    return data
