from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import xml_parser
from sklearn.datasets import make_regression
import random

def TreeRegress(inp):
    arr = xml_parser.xml_parser('TreeRegressor_Params.xml',inp)
    n = len(arr)
    if(n != 15):
        return False
    print(arr)
    n_model = arr[0]
    rng = np.random.RandomState(1)
    # value for max depth
    if(arr[5]!=None):
        arr[5] = 2*np.random.randint(1, arr[1])
    if(arr[7] == float(int(arr[7]))):
        arr[7] = int(arr[7])
    # value for max_features
    if(arr[9]=='int'):
        arr[9] = np.random.randint(1, arr[1])
    elif arr[9] == 'float':
        arr[9] = random.uniform(0, 1)*arr[1]
    elif arr[9] == 'None':
        arr[9] = None
    if(arr[10]=='None'):
        arr[10] = None

    try:
        train_X, train_y = make_regression(n_samples=arr[0],n_features=arr[1],
            n_informative=arr[2])
        print("done1")
    except ValueError:
        print("error1")
        return False
    try:
        random_forest = RandomForestRegressor(n_estimators=arr[3], criterion=arr[4],
            max_depth=arr[5], min_samples_split=arr[6], min_samples_leaf=arr[7],
            min_weight_fraction_leaf=arr[8],max_features=arr[9],
            max_leaf_nodes=arr[10],min_impurity_decrease=arr[11],
            bootstrap=arr[12],oob_score=arr[13], warm_start=arr[14])
        print("done2")
    except ValueError:
        print("error2")
        return False
    try:
        random_forest.fit(train_X, train_y)
    except ValueError:
        print("error3")
        return False
    return True

inp = ["1000 2 2 5 1 0 2 2 0.00001 3 0 2 0 1 0"]
TreeRegress(inp[0])
