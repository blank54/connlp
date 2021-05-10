#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)
from connlp.preprocess import EnglishTokenizer, Normalizer
tokenizer = EnglishTokenizer()
normalizer = Normalizer()


## Preprocess
# Normalizer
def test_normalizer():
    global docs
    
    for doc in docs:
        normalized_doc = normalizer.normalize(text=doc)
        print(normalized_doc)

# English Tokenizer
def test_english_tokenizer():
    global docs
    
    for doc in docs:
        tokenized_doc = tokenizer.tokenize(text=doc)
        print(tokenized_doc)


## Run
if __name__ == '__main__':
    docs = ['I am a boy!', 'He is a boy..', 'She is a girl?']

    test_normalizer()
    test_english_tokenizer()