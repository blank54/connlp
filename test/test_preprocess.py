#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)
from connlp.preprocess import EnglishTokenizer, Normalizer


## Preprocess
# Normalizer
def test_normalizer(docs):
    normalizer = Normalizer()
    for sent in docs:
        normalized_sent = normalizer.normalize(text=sent)
        print(normalized_sent)

# English Tokenizer
def test_english_tokenizer(docs):
    tokenizer = EnglishTokenizer()
    for sent in docs:
        tokenized_sent = tokenizer.tokenize(text=sent)
        print(tokenized_sent)


## Run
if __name__ == '__main__':
    docs = ['I am a boy!', 'She is a girl']

    test_normalizer(docs=docs)
    test_english_tokenizer(docs=docs)