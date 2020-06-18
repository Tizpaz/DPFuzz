from sklearn.utils import gen_batches
import xml_parser

def generate_batches(inp):
    arr = xml_parser.xml_parser('gen_batches_Params.xml',inp)
    n = len(arr)
    if n != 3:
        return False
    print(arr)
    try:
        list(gen_batches(arr[0], arr[1]))
    except ValueError:
        return False

generate_batches("10 1 0")
