import numpy as np
import os
from six.moves import cPickle as pickle
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    # Loads a single batch of CIFAR-10 data
    with open(filename, 'rb') as f: # Open the file in read binary mode
        data_dictionary = load_pickle(f) # Load the data from the file
        X = data_dictionary['data']
        Y = data_dictionary['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    # Load all of cifar data
    xs = [] # List to store X data
    ys = [] # List to store Y data
    for batch in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (batch, )) # Path to the batch file
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs) # X training data
    Ytr = np.concatenate(ys) # Y training data
    del X, Y # Delete X and Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch')) # Load test data
    return Xtr, Ytr, Xte, Yte