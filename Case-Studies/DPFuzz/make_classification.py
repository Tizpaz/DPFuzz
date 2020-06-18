from sklearn.datasets import make_multilabel_classification
import sys
import xml_parser
import signal
import time

def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("FUBAR")

def generate_data_classification(inp):
    arr = xml_parser.xml_parser('make_classification_Params.xml',inp)
    n = len(arr)
    if n != 10:
        return False

    if arr[7] == 'False':
        arr[7] = False

    if arr[9] == 'None':
         arr[9] = None
    else:
        arr[9] = 2


    print(arr)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

    try:
        X, y = make_multilabel_classification(n_samples=arr[0], n_features=arr[1],
                n_classes=arr[2], n_labels=arr[3], length=arr[4],
                allow_unlabeled=arr[5], sparse=arr[6], return_indicator=arr[7],
                return_distributions=arr[8], random_state=arr[9])
    except ValueError as err:
        print("value error: ")
        print(sys.exc_info())
        return False
    except ZeroDivisionError as err:
        print("zero division error: ")
        print(err)
        return False
    except:
        print("Unexpected error:")
        print(sys.exc_info()[0])
        raise
        return False
    finally:
        signal.alarm(0)
        print("After: %s" % time.strftime("%M:%S"))
        return True
    return True


generate_data_classification("1000 5 0 0 50 1 1 1 1 0")
