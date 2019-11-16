import numpy as np
def get_outliers(data,  std=2):
    mean = np.mean(data)
    std = np.std(data)
    outliers = [x for x in data if x > mean + 2*std] + [ x for x in data if x < mean - 2*std]
    return outliers


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(probs):
    return [probs[i]/sum(probs) for i in probs ]

