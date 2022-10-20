import numpy as np

def replace_zeroes(data):
    min_nonzero = np.min(np.abs(data[np.nonzero(data)]))
    print(f'min_nonzero: {min_nonzero}')
    data[data == 0] = min_nonzero
    # data[data == 0] = 0.00001
    return data
