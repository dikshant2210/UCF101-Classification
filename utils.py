import numpy as np


def rescale_list(array, size):
    length = len(array)
    if length < size:
        return None
    skip = length // size
    output = np.array([array[i] for i in range(0, length, skip)])
    return output[0:size]
