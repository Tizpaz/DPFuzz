import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import xml_parser
import random

def DecisionTree(inp):
    arr = xml_parser.xml_parser('Decision_Tree_Classifier_Params.xml',inp)
    n = len(arr)
    if n != 18:
        return False
    try:
        X, y = make_classification(n_samples=arr[0], n_features=arr[1],
                n_informative=arr[2], n_classes=arr[3])
    except ValueError:
        # print("here")
        return False
    print(arr)

    if(arr[6] == 'None'):
        arr[6] = None
    else:
        arr[6] = random.randint(1, 100)

    if arr[7] == int(arr[7]):
         arr[7] = int(arr[7])
    else:
        arr[7] = arr[7]/100.0

    if arr[8] == int(arr[8]):
         arr[8] = int(arr[8])
    else:
        arr[8] = arr[8]/50.0

    if arr[10] == 'val':
        if arr[11] == int(min(arr[11],arr[1])):
             arr[11] = int(arr[11])
        else:
            arr[11] = arr[11]/10.0
    elif arr[10] == 'None':
        arr[11] = None
    else:
        arr[11] = arr[10]

    if arr[12] == 'None':
         arr[12] = None
    else:
        arr[12] = random.randint(1, 10)

    if arr[13] == 'None':
         arr[13] = None
    else:
        arr[13] = random.randint(1, 100)

    if arr[16] == 'None':
         arr[16] = None
    elif arr[16] == 'weighted':
        weight_lst = {}
        for class_num in range(arr[3]):
            weight_lst[class_num] = random.randint(1, 5)
        arr[16] = weight_lst


    # X = StandardScaler().fit_transform(X)
    try:
        clf = DecisionTreeClassifier(criterion=arr[4], splitter=arr[5], max_depth=arr[6],
                min_samples_split=arr[7], min_samples_leaf=arr[8], min_weight_fraction_leaf=arr[9],
                max_features=arr[11], random_state=arr[12], max_leaf_nodes=arr[13],
                min_impurity_decrease=arr[14], class_weight=arr[16],
                presort=arr[17])
        clf.fit(X, y)
        print("here1")
    except ValueError:
        print("here2")
        return False
    # except KeyError:
    #     # print("here3")
    #     return False
    return True

inp = ['1000 2 2 2 1 1 0 10 10 0 0 1 0 1 0 1 2 1']
DecisionTree(inp[0])
