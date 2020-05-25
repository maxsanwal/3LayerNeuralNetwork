# imports
import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input data
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# output data
y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)
# weights
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# training
for j in range(60000):

    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # l2 error
    l2_error = y - l2

    if (j % 10000) == 0:
        print('Error', np.mean(np.abs(l2_error)))

    l2_delta = l2_error * nonlin(l2, deriv=True)

    # l1 error
    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update weights
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
