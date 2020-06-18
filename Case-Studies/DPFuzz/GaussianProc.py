import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_classification
import xml_parser

def GaussianProcess(inp):
    arr = xml_parser.xml_parser('Gaussian_Proc_Params.xml',inp)
    n = len(arr)
    if n != 14:
        return False
    try:
        X, y = make_classification(n_samples=arr[0], n_features=arr[1],
                n_informative=arr[2], n_classes=arr[3])
    except ValueError:
        # print("here")
        return False
    print(arr)
    kernel = arr[4] * RBF([1.0 for i in range(arr[5])])

    if(arr[6] == 'None'):
        arr[6] = None

    if arr[11] == 'None':
         arr[11] = None
    else:
        arr[11] = 2
    # X = StandardScaler().fit_transform(X)
    try:
        clf = GaussianProcessClassifier(kernel=kernel, optimizer=arr[6],
                n_restarts_optimizer=arr[7], max_iter_predict=arr[8], warm_start=arr[9],
                copy_X_train=arr[10], random_state=arr[11], multi_class=arr[12], n_jobs=arr[13])
        clf.fit(X, y)
        # print("here1")
    except ValueError:
        # print("here2")
        return False
    # except KeyError:
    #     # print("here3")
    #     return False
    return True

inp = ['1000 10 2 2 1 1 1 10 1 1 1 1 1 1']
GaussianProcess(inp[0])
