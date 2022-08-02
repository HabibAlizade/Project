from colorama import Fore
print(Fore.BLUE)
import numpy as np
import pandas as pd
import gzip
import pickle


def regularize(data):
    data_input = [x.reshape(len(x), 1) for x in data[0]]
    data_output = [vectorized(y) for y in data[1]]
    return list(zip(data_input, data_output))

  
def vectorized(y):
  v = np.zeros((10,1))
  v[y] = 1.0
  return v


with gzip.open('../NeuralNetworks-Cloned/data/mnist.pkl.gz', 'rb') as f:
    train, validation, test = pickle.load(f, encoding='latin1')
#Equivalently:
# u = pickle._Unpickler(f)
# u.encoding = 'latin1'
# train, validation, test  = u.load()

train_data = regularize(train)
validation_data = regularize(validation)
test_data = regularize(test)
