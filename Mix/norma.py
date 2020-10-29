import numpy as np


def norm(v):
    if type(v) is not np.ndarray:
        raise RuntimeError('Please provide a numpy array.')
    elif np.ndim(v) > 2:
        raise RuntimeError('Too many dimensions!')
    elif np.ndim(v) == 2 and np.shape(v)[0] != 1 and np.shape(v)[1] != 1:
        raise RuntimeError('This is not a numpy vector.')
    cumsum = 0
    
    flatv = v.flatten()
    
    for element in flatv:
        cumsum += element**2

    return np.sqrt(cumsum)


# print(norm(np.array([[1],[2],[3]])))
# print(norm(np.array([[1,2,3]])))
# print(norm(np.array([1,2,3])))