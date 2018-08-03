import numpy as np
import tables
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X_MAX, Y_MAX, Z_MAX = 191., 191., 191.
E_SCALE = 50


def plot_3d(X, l):
    """
    Function that draw two 3D-plots. First one with energy as a color, and second one shows labels.
    :param X: np.array[N, 4]
    :param l: np.array[N]
    :return: None
    """
    fig = plt.figure(1, figsize=(20, 20))
    
    x, y, z = X.T[:3, :]
    if X.shape[1] > 3:
        v = X[:, 3]
    else:
        v = np.zeros(len(X))
    
    if len(l.shape) > 1:
        label = np.argmax(l, axis=1)
    else:
        label = l
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(x,y,z,c=v, marker='o', vmin=0, vmax=1)
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(x,y,z,c=label, marker='o')
    plt.show()


def ohe_transform(y, num_classes=2):
    if num_classes == 2:
        y_ohe = np.zeros((len(y), 2))
        y_ohe[:, 0] = (y == 1.)
        y_ohe[:, 1] = (y == 2.)
    elif num_classes == 4:
        y_ohe = np.zeros((len(y), 4))
        y_ohe[:, 0] = (y == 0.)
        y_ohe[:, 1] = (y == 1.)
        y_ohe[:, 2] = (y == 2.)
        y_ohe[:, 3] = (y == 3.)
    else:
        raise BaseException
    return y_ohe


def hdf5_to_numpy(file='./sample3d.hdf5', n=100, num_classes=2, test=False):
    """
    Converts .hdf5-data to numpy array.
    :param file: string
    :param n: int or np.inf
    :param num_classes: int, either 2 or 3
    :return:
    X: np.array[N, 4] -- x, y, z coordinates, energy normalized to [0, 1] range
    Y: np.array[N] -- label
    M: np.array[N, 3] -- original positions of each entry
    N: int -- total number of events
    """
    f = tables.open_file(file)
    X = list()
    Y = list()
    M = list()
    N = min(len(f.root.data), n)
    print(len(f.root.data))
    for i in tqdm(range(N)):
        data = f.root.data[i]
        mask = data > 0
        x, y, z = np.where(mask)
        v = data[np.where(mask)]
        
        X.append(np.stack([x / X_MAX, 
                           y / Y_MAX,
                           z / Z_MAX,
                           v / E_SCALE]).T)
        if not test:
            label = f.root.label[i]
            Y.append(
                ohe_transform(label[np.where(mask)], num_classes=num_classes)
            )
        else:
            Y.append(np.zeros((len(np.where(mask)[0]), num_classes)))
        M.append(np.array(np.where(mask)).T)
    return X, Y, M, N
