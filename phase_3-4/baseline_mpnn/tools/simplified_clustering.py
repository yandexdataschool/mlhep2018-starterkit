import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
EPS = 1e-17


def generate_edges(X, mode='kneighbors_graph', n_neighbors=3, radius=0.1):
    """
    returns array with pairs of indices [vertex_from, vertex_to] and weight vector
    """
    n_neighbors = min(n_neighbors, len(X) - 1)
    if mode == 'kneighbors_graph':
        adjacency_matrix = np.array((kneighbors_graph(X=X[:, :3], 
                                                      n_neighbors=n_neighbors, mode='distance')).todense())
    elif mode == 'radius_neighbors_graph':
        adjacency_matrix = np.array((radius_neighbors_graph(X=X[:, :3], 
                                                            radius=radius, mode='distance')).todense())
    else:
        raise 'Unknown mode {}'.format(mode)
    rows, cols = np.where(adjacency_matrix > 0)
    edges = np.vstack([rows, cols])
    weights = adjacency_matrix[rows, cols]
    
    nodes_features = X[:, 3].reshape(-1, 1)    
    edges_features = X[edges.T[:, 0]] - X[edges.T[:, 1]]
                
    return nodes_features, np.c_[edges_features, weights], edges.astype(int)


def generate_graph_dataset(X, Y, M, n_neighbors, radius=0.1, in_degree_max=None, out_degree_max=None, mode='kneighbors_graph'):
    X_graph_dataset = {}
    
    N = len(X)
    nodes_features, edges_features, edges_in_out = generate_edges(X, n_neighbors=n_neighbors, mode=mode, radius=radius)
    X_graph_dataset['X_cluster_nodes'] = nodes_features
    X_graph_dataset['X_cluster_edges'] = edges_features
    X_graph_dataset['X_cluster_in_out'] = edges_in_out.T
        
    X_cluster_messages_in = [[] for _ in range(N)]
    for i, value in enumerate(edges_in_out[1, :]):
        X_cluster_messages_in[int(value)].append(i)

    X_cluster_messages_out = [[] for _ in range(N)]
    for i, value in enumerate(edges_in_out[0, :]):
        X_cluster_messages_out[int(value)].append(i)
        
    last_index = len(edges_in_out.T)
    if in_degree_max is None:
        in_degree_max = 0
        for row in X_cluster_messages_in:
            in_degree_max = max(in_degree_max, len(row))
    for row in X_cluster_messages_in:
        if len(row) >= in_degree_max:
            row[:] = row[:in_degree_max]
        else:
            row += (in_degree_max - len(row)) * [last_index]

    if out_degree_max is None:
        out_degree_max = 0
        for row in X_cluster_messages_out:
            out_degree_max = max(out_degree_max, len(row))

    for row in X_cluster_messages_out:
        if len(row) >= out_degree_max:
            row[:] = row[:out_degree_max]
        else:
            row += (out_degree_max - len(row)) * [last_index]

            
    X_graph_dataset['X_cluster_messages_in'] = np.array(X_cluster_messages_in)
    X_graph_dataset['X_cluster_messages_out'] = np.array(X_cluster_messages_out)


    assert X_graph_dataset['X_cluster_messages_in'].shape[1] == in_degree_max
    assert X_graph_dataset['X_cluster_messages_out'].shape[1] == out_degree_max

    X_graph_dataset['Y_cluster_labels'] = Y
    X_graph_dataset['M'] = M

    return X_graph_dataset, in_degree_max, out_degree_max