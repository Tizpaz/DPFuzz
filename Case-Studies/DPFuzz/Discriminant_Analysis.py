from scipy import linalg
import numpy as np
import xml_parser
from sklearn.datasets import make_classification

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

def disc_analysis(inp):
    arr = xml_parser.xml_parser('Discriminant_Analysis_Params.xml',inp)
    n = len(arr)
    if n != 13:
        return False
    try:
        # if(arr[5]=='eigen' and arr[1] > 3):
        #     arr[1] = 3
        # if(arr[5]=='lsqr' and arr[1] > 3):
        #     arr[1] = 3
        # X, y = make_classification(n_samples=arr[0], n_features=arr[1],
        #             n_informative=arr[2], n_classes=arr[3])
        if(arr[12]==0):
            X, y = dataset_fixed_cov(arr[0],arr[1])
        elif(arr[12]==1):
            X, y = dataset_cov(arr[0],arr[1])
            # print("here!!")
        else:
            if(arr[5]=='svd'):
                X, y = make_classification(n_samples=arr[0], n_features=arr[1],
                                        n_informative=arr[2], n_classes=arr[3])
            else:
                return False
        # print("done1")
    except ValueError:
        # print("error1")
        return False
    # print("here")

    # value for parameter_6
    if(arr[6]=="float"):
        arr[6] = random.uniform(0, 1)
    elif(arr[6]=="auto"):
        arr[6] = "auto"
    else:
        arr[6] = None
    # value for parameter_7
    # if(arr[7]!=None):
    #     val_7 = np.random.dirichlet(np.ones(arr[3]),size=1.0)
    # else:
    arr[7] = None
    # value for parameter_8
    if(arr[8]!='None'):
        arr[8] = np.random.randint(1,arr[3])
    else:
        arr[8] = None

    # Note in sklearn page
    if(arr[5]=='svd' and arr[6] != None):
        return False

    print(arr)
    if(arr[4]):
        try:
            # Linear Discriminant Analysis
            lda = LinearDiscriminantAnalysis(solver=arr[5],
            shrinkage=arr[6],priors=arr[7],n_components=arr[8],
            store_covariance=arr[9], tol=arr[10])
            # print("done2")
        except ValueError:
            # print("error2")
            return False
        try:
            y_pred = lda.fit(X, y)
            # print("done3")
        except TypeError:
            # print("error3")
            return False
    else:
        try:
            # Quadratic Discriminant Analysis
            qda = QuadraticDiscriminantAnalysis(priors=arr[7],
                reg_param=arr[11],store_covariance=[9], tol=arr[10])
            # print("here21")
        except ValueError:
            return False
        try:
            y_pred = qda.fit(X, y)
            # print("here22")
        except TypeError:
            return False

# inp = ["1000 2 2 2 1 1 0 1 0 0 0.01 0.0 1"]
# disc_analysis(inp[0])
