import numpy as np
import tables
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

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


def ohe_transform(y, num_classes=2, zero_class_exists=False):
    if zero_class_exists:
        y_ohe = np.zeros((len(y), num_classes))
        for i in range(num_classes):
            y_ohe[:, i] = (y == i)
    else:
        y_ohe = np.zeros((len(y), num_classes - 1))
        for i in range(num_classes - 1):
            y_ohe[:, i] = (y == i + 1.)
    return y_ohe


def hdf5_to_numpy(file='./sample3d.hdf5', n=100, num_classes=2, test=False, zero_class_exists=False, energy_cut=0.01):
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
        mask = data >= energy_cut
        x, y, z = np.where(mask)
        v = data[np.where(mask)]
        
        X.append(np.stack([x / X_MAX, 
                           y / Y_MAX,
                           z / Z_MAX,
                           v / E_SCALE]).T)
        if not test:
            label = f.root.label[i]
            Y.append(
                ohe_transform(label[np.where(mask)], num_classes=num_classes, zero_class_exists=zero_class_exists)
            )
        else:
            Y.append(np.zeros((len(np.where(mask)[0]), num_classes - (not zero_class_exists))))
        M.append(np.array(np.where(mask)).T)
    return X, Y, M, N



def csv_to_numpy(file, num_classes=3, energy_cut=0.01, zero_class_exists=True, test=False):
    from copy import deepcopy
    df = pd.read_csv(file, delimiter=',')
    df['x'] /= X_MAX
    df['y'] /= Y_MAX
    df['z'] /= Z_MAX
    df['val'] /= E_SCALE
    
    grouped_df = df.groupby('event')
    
    X = []
    Y = []
    M = []
    events = []
    
    for event, group in tqdm(grouped_df):
        events.append(event)
        mask = group.val > energy_cut / E_SCALE
        X.append(group[mask][['x', 'y', 'z', 'val']].values)
        M.append(mask)
        if not test:
            Y.append(ohe_transform(group[mask]['label'], num_classes=num_classes, zero_class_exists=zero_class_exists))
        else:
            Y.append(np.zeros((mask.sum(), num_classes - (not zero_class_exists))))
            
    return X, Y, M, events


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
    
def plot_3d_with_edges(X, l, edges, azim=-84, elev=10):
    x, y, z = X.T[:3, :]
    if X.shape[1] > 3:
        v = X[:, 3]
    else:
        v = np.zeros(len(X))
    
    if len(l.shape) > 1:
        label = np.argmax(l, axis=1)
    else:
        label = l
    
    x_start = X[edges[:, 0], :][:, 0]
    x_end = X[edges[:, 1], :][:, 0]
    
    y_start = X[edges[:, 0], :][:, 1]
    y_end = X[edges[:, 1], :][:, 1]
    
    z_start = X[edges[:, 0], :][:, 2]
    z_end = X[edges[:, 1], :][:, 2]

    
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    z_min, z_max = X[:, 2].min(), X[:, 2].max()
    
    # plot only hits
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=azim, elev=elev)
    ax.scatter(x,y,z, c='k', s=2)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_zlabel("x")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    plt.show()

    # plot with labels 
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=azim, elev=elev)
    ax.scatter(x,y,z,c=np.log(label + 1), s=2)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_zlabel("x")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    plt.show()
    
    # plot edges
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=azim, elev=elev)
    start_points = np.array([x_start, y_start, z_start]).T.reshape(-1, 3)
    end_points = np.array([x_end, y_end, z_end]).T.reshape(-1, 3)
    C = plt.cm.Blues(0.9)
    lc = Line3DCollection(list(zip(start_points, end_points)), colors=C, alpha=0.007,lw=2)
    ax.add_collection3d(lc)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_zlabel("x")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    plt.show()    
