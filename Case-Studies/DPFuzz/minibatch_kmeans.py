import numpy as np
# import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
import xml_parser
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Generate datasets
def dataset_fixed_cov(n, dim):
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    # n, dim = 300, 2
    np.random.seed(0)
    # C = np.array([[0., -0.23], [0.83, .23]])
    # C = np.random.rand()
    X = np.r_[np.random.randn(n, dim),
              np.random.randn(n, dim) + 1]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def dataset_cov(n, dim):
    '''Generate 2 Gaussians samples with different covariance matrices'''
    # n, dim = 300, 2
    np.random.seed(0)
    # C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.random.randn(n, dim),
              np.random.randn(n, dim) * 0.3]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

def make_moon(n_samples):
    print("here_moon")
    X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    return X, y

def make_circles(n_samples):
    print("here_circ")
    X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    return X, y


def minibatch_kmeans(inp):
    arr = xml_parser.xml_parser('minibatch_kmeans_Params.xml',inp)
    n = len(arr)
#    print(n)
    if n != 15:
        return False
    try:
        if(arr[2]==0):
            X, y = make_classification(n_samples=arr[0], n_features=arr[1],
                    n_informative=2, n_classes=arr[3])
        elif(arr[2]==1):
            X, y = dataset_fixed_cov(arr[0],arr[1])
        elif(arr[2]==2):
            X, y = dataset_cov(arr[0],arr[1])
        # else:
        #     X, y = make_circles(arr[0])
    except ValueError:
        return False
    print(arr)
    X = StandardScaler().fit_transform(X)

    if arr[9] == 'None':
         arr[9] = None
    else:
        arr[9] = 2

    if arr[12] == 0:
         arr[12] = None
    elif arr[12] < arr[3]:
        arr[12] = arr[3]


    try:
        MBKM = cluster.MiniBatchKMeans(n_clusters=arr[3], init=arr[4],
            max_iter=arr[5], batch_size=arr[6], verbose=arr[7], compute_labels=arr[8],
            random_state=arr[9], tol=arr[10], max_no_improvement=arr[11],
            init_size=arr[12], n_init=arr[13], reassignment_ratio=arr[14])
        MBKM.fit(X)
        print("Done!")
    except ValueError:
        return False
inp = ['10000 2 2 2 1 0 0.001 0 1 0 0.01 10 2 1 0.1']
minibatch_kmeans(inp[0])
