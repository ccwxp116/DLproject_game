import tensorflow as tf
import numpy as np
import preprocess
from types import SimpleNamespace
import rnn

# get the data
train_id, test_id, vocab = preprocess.get_data("../data/nlp_train.txt", "../data/nlp_test.txt")
train_id = np.array(train_id)
test_id  = np.array(test_id)
X0, Y0 = train_id[:-1], train_id[1:]
X1, Y1  = test_id[:-1],  test_id[1:]
print(X0.shape, Y0.shape)

def process_data(window_size, data):
    remainder = (len(data) - 1)%window_size
    data = data[:-remainder]
    data = data[:-1].reshape(-1, 20)
    return data

X0 = process_data(20, X0)
Y0 = process_data(20, Y0)
X1 = process_data(20, X1)
Y1 = process_data(20, Y1)
print(X0.shape, Y0.shape)

rnn_args = rnn.get_text_model(vocab)

rnn_args.model.fit(
    X0, Y0,
    epochs=rnn_args.epochs, 
    batch_size=rnn_args.batch_size,
    validation_data=(X1, Y1)
)