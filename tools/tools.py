from sklearn import metrics
import numpy as np


def stretch_array(X, n, fill_value=0):
    """
    :param X:
    :param n:
    :return:
    """
    if X.shape[1] >= n:
        return X[:, :n]
    else:
        X_new = np.full((len(X), n), fill_value=fill_value)
        X_new[:, :X.shape[1]] = X
        return X_new


def calculate_metrics_on_graph(X_cluster_graph, predictions):
    """
    return: accuracy, roc_auc
    """
    y_ravel = []
    predictions_ravel = []
    for cluster, prediction_global in zip(X_cluster_graph['clusters'], predictions):
        y_ravel += (cluster['Y'].tolist())
        predictions_ravel += len(cluster['Y']) * [prediction_global]
    y_ravel = np.array(y_ravel)
    predictions_ravel = np.array(predictions_ravel)

    accuracy = metrics.accuracy_score(np.argmax(y_ravel, axis=1), np.argmax(predictions_ravel, axis=1))
    try: roc_auc = metrics.roc_auc_score(y_ravel, predictions_ravel)
    except: roc_auc = 0.5
    return accuracy, roc_auc, predictions_ravel, y_ravel
    
def prepare_data_for_submission(X_cluster_graph):
    submission_answer = np.zeros((192, 192, 192))
    if X_cluster_graph['empty']:
        pass
    else:
        for cluster, prediction in zip(X_cluster_graph['clusters'], X_cluster_graph['predictions']):
            submission_answer[cluster['M'][:, 0], cluster['M'][:, 1], cluster['M'][:, 2]] = np.argmax(prediction) + 1
    X_cluster_graph['submission_answer'] = submission_answer