from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
import pickle
import numpy as np
import time
import tensorflow as tf
from transformer_utils import LookAheadMaskLayer, generate_text

######################################################################################################
# LOAD MODELS
######################################################################################################
# Text model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Path to the .h5 file
text_model_path = '../model/transformer_batch32_epoch1_embedding64_headsize128_numhead2_dropout0.5_lr1e-4_separatepunc_5k.h5'

# Load the model
text_model = load_model(text_model_path, 
                   custom_objects={'LookAheadMaskLayer': LookAheadMaskLayer})
text_model.summary()

# image model


######################################################################################################
# GENERATE TEXT
######################################################################################################
input_text = input("Please enter a brief starting text: ")
output_text = generate_text(text_model, tokenizer, input_text)
print('-------------------Generated Text Below-------------------')
print(output_text)

######################################################################################################
# GENERATE TEXT
######################################################################################################
