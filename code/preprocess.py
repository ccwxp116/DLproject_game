import tensorflow as tf
import numpy as np
from functools import reduce
import random
import pandas as pd
import re


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg index->word mapping)
    """
    # Hint: You might not use all of the initialized variables depending on how you implement preprocessing. 
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    with open(train_file, 'r', encoding='utf-8') as file:
        for line in file:
            sentence_list = line.lower().split()
            for word in sentence_list:
                train_data.append(word)
    
    # built vocab list
    train_unique_words = sorted(set(train_data))
    vocabulary = {w:i for i, w in enumerate(train_unique_words)}
    vocabulary['<UNK>'] = len(vocabulary)

    with open(test_file, 'r', encoding='utf-8') as file:
        for line in file:
            sentence_list = line.lower().split()
            for word in sentence_list:
                if word in vocabulary:  
                    test_data.append(word)
                else:
                    test_data.append('<UNK>')
    
    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)

    # Vectorize, and return output tuple.
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    return train_data, test_data, vocabulary

# split train and test data

def split_train_test(data):
    size = data.shape[0]
    train_size = int(size * 0.8)
    train_indices = random.sample(range(size), train_size)
    train_data = data.iloc[train_indices]

    all_indices = np.arange(data.shape[0])
    test_indices = np.setdiff1d(all_indices, train_indices)
    test_data = data.iloc[test_indices]

    train_data.to_csv('../data/train.csv', index=True)
    test_data.to_csv('../data/test.csv', index=True)

    print('<Done splitting train and test!>')

    return train_data, test_data

# separate punctuation
def separate_punctuation(text):
    # Use regex to find punctuations and add space around them
    processed_text = re.sub(r"([,.!?:'-0123456789#$%&()~])", r" \1 ", text)
    # Remove extra spaces
    processed_text = re.sub(r"\s{2,}", " ", processed_text)
    return processed_text.strip()

# RUN test the function
# file_path = "../data/clean_data.csv"
file_path = "../data/data.csv"
data = pd.read_csv(file_path)

data['About the game'] = data['About the game'].apply(separate_punctuation)
print("<DataFrame processed with separated punctuation>")
data['About the game'] = data['About the game'].replace('"', '')
data = data[['About the game', 'Genres']]

# split the data
train, test = split_train_test(data)

# for classification
train.to_csv('../data/train_class.csv', index=False, header=True)
test.to_csv('../data/test_class.csv', index=False, header=True)

# for each train and test, only reserve the text data
nlp_train = train['About the game'].replace('"', '')
nlp_test = test['About the game'].replace('"', '')
# save the train and test for MLP
nlp_train.to_csv('../data/nlp_train.txt', sep='\t', index=False, header=False)
nlp_test.to_csv('../data/nlp_test.txt', sep='\t', index=False, header=False)
print('<nlp train and test exported>')

