#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)
from connlp.preprocess import EnglishTokenizer
from connlp.visualize import Visualizer


## Visualize
# Word Network
def test_word_network(docs):
    tokenizer = EnglishTokenizer()
    tokenized_docs = [tokenizer.tokenize(text=doc) for doc in docs]

    visualizer = Visualizer()
    visualizer.network(docs=tokenized_docs, show=True)


## Run
if __name__ == '__main__':
    docs = ['I am a boy', 'She is a girl']
    test_word_network(docs=docs)