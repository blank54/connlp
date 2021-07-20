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
from connlp.util import makedir


## Tokenize
def tokenize_docs(docs):
    tokenizer = EnglishTokenizer()
    tokenized_docs = [tokenizer.tokenize(text=doc) for doc in docs]
    return tokenized_docs

## Visualize
# Word Network
def test_word_network(tokenized_docs):
    visualizer = Visualizer()
    wnt = visualizer.network(docs=tokenized_docs, show=False)

    fname = 'wordnetwork.png'
    fpath = os.path.join('test/wordnetwork/', fname)
    makedir(fpath)
    wnt.savefig(fpath, dpi=300, bbox_inches='tight', pad_inches=0)

# Word Cloud
def test_wordcloud(tokenized_docs):
    visualizer = Visualizer()
    wcd = visualizer.wordcloud(docs=tokenized_docs, show=False)

    fname = 'wordcloud.png'
    fpath = os.path.join('test/wordcloud/', fname)
    makedir(fpath)
    wcd.savefig(fpath, dpi=300, bbox_inches='tight', pad_inches=0)


## Run
if __name__ == '__main__':
    ## Tokenize docs
    docs = ['I am a boy', 'She is a girl']
    tokenized_docs = tokenize_docs(docs=docs)

    ## Visualize
    test_word_network(tokenized_docs=tokenized_docs)
    test_wordcloud(tokenized_docs=tokenized_docs)