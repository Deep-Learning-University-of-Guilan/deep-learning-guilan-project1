import numpy as np


class BinaryCrossEntropy:
    def __call__(self, y_true, y_pred, epsilon=1e-15):
        return -(y_true * (np.log(y_pred + epsilon)) + (1-y_true) * np.log((1 - y_pred) + epsilon))

    def backward(self, y_true, y_pred):
        l = -(y_true / y_pred) + ((1-y_true) / (1-y_pred))
        if np.isnan(l):
            return 0
        elif np.isinf(l):
            return 1
        else:
            return l
