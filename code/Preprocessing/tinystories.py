import tensorflow as tf
import numpy as np
from functools import reduce
import random
import pandas as pd
import re

def read_file_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines

def sample_lines(lines, percentage):
    number_of_lines = int(len(lines) * percentage)
    sampled_lines = random.sample(lines, number_of_lines)
    return sampled_lines

def write_lines_to_file(lines, output_filename):
    with open(output_filename, 'w') as file:
        file.writelines(lines)

# separate punctuation
def separate_punctuation(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Regular expression to match punctuation
    # This regex inserts a space before and after each punctuation mark
    modified_text = re.sub(r"([.,!?;:])", r" \1 ", text)

    # Optionally, you might want to clean up multiple spaces caused by the previous step
    modified_text = re.sub(r'\s{2,}', ' ', modified_text)

    return modified_text

# RUN test the function
file_path = "../data/TinyStories.txt"
output_file_path = '../data/tinystory.txt'
all_lines = read_file_lines(file_path)

sampled_lines = sample_lines(all_lines, 0.76)

write_lines_to_file(sampled_lines, output_file_path)

modified_text = separate_punctuation(output_file_path)

# Write the modified text to a new file
write_lines_to_file(modified_text, output_file_path)


##################################################
def read_and_combine_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # Read lines from each file
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Combine the lists
    combined_lines = lines1 + lines2
    return combined_lines

def shuffle_lines(lines):
    random.shuffle(lines)
    return lines

def write_shuffled_lines_to_file(shuffled_lines, output_file):
    with open(output_file, 'w') as file:
        file.writelines(shuffled_lines)

file_path1 = '../data/nlp_train_genre.txt'
file_path2 = '../data/tinystory.txt'
output_file_path = '../data/combine_game_tinystory.txt'

# Read and combine files
combined_lines = read_and_combine_files(file_path1, file_path2)

# Shuffle lines
shuffled_lines = shuffle_lines(combined_lines)

# Write shuffled lines to a new file
write_shuffled_lines_to_file(shuffled_lines, output_file_path)