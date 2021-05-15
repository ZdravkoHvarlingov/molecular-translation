import pandas as pd
from common.vocabulary import Vocabulary

vocab = Vocabulary()
stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'I': 4, 'n': 5, 'C': 6, 'h': 7, '=': 8, '1': 9, 'S': 10, '/': 11, '3': 12, 'H': 13, '2': 14, '0': 15, 'O': 16, 'c': 17, '-': 18, '9': 19, '(': 20, ')': 21, '8': 22, '5': 23, '6': 24, '7': 25, '4': 26, ',': 27, 't': 28, '+': 29, 'm': 30, 's': 31, 'N': 32, 'B': 33, 'r': 34, 'b': 35, 'F': 36, 'l': 37, 'P': 38, 'i': 39, 'D': 40, 'T': 41}
vocab_stoi = vocab.stoi

words = set(stoi.keys())
vocab_words = set(vocab.stoi.keys())

print(words == vocab_words)