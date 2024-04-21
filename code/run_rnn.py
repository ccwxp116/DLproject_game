import tensorflow as tf
import numpy as np
import preprocess
from types import SimpleNamespace
# from rnn import *

# get the data
train_id, test_id, vocab = preprocess.get_data("../data/nlp_train.txt", "../data/nlp_test.txt")
train_id = np.array(train_id)
test_id  = np.array(test_id)
X0, Y0 = train_id[:-1], train_id[1:]
X1, Y1  = test_id[:-1],  test_id[1:]
print(X0.shape, Y0.shape)