import tensorflow as tf
import numpy as np
import preprocess
from types import SimpleNamespace
import trigram

train_id, test_id, vocab = preprocess.get_data("../data/nlp_train.txt", "../data/nlp_test.txt")
train_id = np.array(train_id)
test_id  = np.array(test_id)
X0, Y0 = train_id[:-1], train_id[1:]
X1, Y1  = test_id[:-1],  test_id[1:]
print(X0.shape, Y0.shape)

def process_trigram_data(data):
    X = np.array(data[:-1])
    Y = np.array(data[2:])
    X = np.column_stack((X[:-1], X[1:]))
    return X, Y

X0, Y0 = process_trigram_data(train_id)
X1, Y1 = process_trigram_data(test_id)

args = trigram.get_text_model(vocab)

args.model.fit(
    X0, Y0,
    epochs=args.epochs, 
    batch_size=args.batch_size,
    validation_data=(X1, Y1)
)