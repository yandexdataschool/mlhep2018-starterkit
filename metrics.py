import numpy as np

def acc_at_80(accuraces):
    return np.percentile(accuraces, 80)

def acc_at_50(accuraces):
    return np.median(accuraces)

def acc_mean(accuraces):
    return np.mean(accuraces)
