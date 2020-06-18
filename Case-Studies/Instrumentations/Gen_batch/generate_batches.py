# from sklearn.utils import gen_batches
import xml_parser

def gen_batches(n, batch_size, min_batch_size=0):
    """Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.
    Parameters
    ----------
    n : int
    batch_size : int
        Number of element in each batch
    min_batch_size : int, default=0
        Minimum batch size to produce.
    Yields
    ------
    slice of batch_size elements
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

def generate_batches(inp):
    arr = xml_parser.xml_parser('gen_batches_Params.xml',inp)
    n = len(arr)
    if n != 3:
        return False
#    print(arr)
    try:
        list(gen_batches(arr[0], arr[1]))
    except ValueError:
        return False

# generate_batches("10000 1.0 0")
