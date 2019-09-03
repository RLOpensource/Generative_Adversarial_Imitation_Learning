import numpy as np

def list_to_onehot(input, action_size):
    result = []
    for i in input:
        x = np.zeros(action_size)
        x[i] = 1
        result.append(x)
    return np.stack(result)