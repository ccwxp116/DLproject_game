from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import preprocess
import pickle
import numpy as np
import time

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def acc_generation(model, text):

    for i in range(15):
    # Tokenize the input text
        token_text = tokenizer.texts_to_sequences([text])[0]
        # Pad the tokenized text
        padded_token_text = pad_sequences([token_text], maxlen=1041, padding='pre')
        # Predict the next word index
        pos = np.argmax(model.predict(padded_token_text))

        # Retrieve the word corresponding to the predicted index
        for word, index in tokenizer.word_index.items():
            if index == pos:
                # Append the predicted word to the input text
                text = text + " " + word
                print(text)
                # Simulate typing effect with a delay of 1 second
                time.sleep(1)
    return text


# Path to the .h5 file
model_path = '100_150_epoch3_model.h5'

# Load the model
model = load_model(model_path)

# print the sentence generation
text = 'When the Roman people honor a simple warrior' #'The army now has a new robotics project'
text_generation = acc_generation(model, text)
print('>>> ', text_generation)

