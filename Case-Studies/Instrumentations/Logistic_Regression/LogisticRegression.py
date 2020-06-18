import itertools
import time
import xml_parser

import numpy as np
from sklearn.externals.joblib import parallel_backend
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.datasets import make_classification

def logistic_regression(inp):
    arr = xml_parser.xml_parser('logistic_regression_Params.xml',inp)
    n = len(arr)
    if n != 14:
        return False
    if arr[5] != 'newton-cg':
        return False

    try:
        X, y = make_classification(n_samples=arr[0], n_features=arr[1],
                n_informative=arr[2], n_classes=arr[3])
    except ValueError:
        # print("here")
        return False
    t = 0
    if arr[8] == 0.0 and arr[13] == 'multinomial' and arr[6] == 'l2':
        t = 1
    else:
        t = 2

    try:
        clf = LogisticRegression(penalty=arr[6], dual = arr[7], tol = arr[8],
        C = arr[9], fit_intercept = arr[10], intercept_scaling = arr[11],
        solver=arr[5], n_jobs=arr[4], max_iter = arr[12], multi_class=arr[13])
        clf.fit(X, y)
        # print("here1")
    except ValueError:
        # print("here2")
        return False
    except IOError:
        return False
    # except KeyError:
    #     # print("here3")
    #     return False
    return True

# inp = ["10000 10 2 2 1 2 1 1 0.0001 1.0 0 1.0 100 0"]
# logistic_regression(inp[0])
