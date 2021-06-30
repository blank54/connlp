#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)
from connlp.preprocess import EnglishTokenizer, Normalizer, KoreanTokenizer
eng_tokenizer = EnglishTokenizer()
normalizer = Normalizer()
kor_tokenizer = KoreanTokenizer(min_frequency=0)

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
        tokenized_doc = eng_tokenizer.tokenize(text=doc)
        print(tokenized_doc)

# Korean Tokenizer
def test_korean_tokenizer(kor_docs):
    print('\nTRAINING Korean Tokenizer...')
    kor_tokenizer.train(kor_docs)

    print()

    for doc in kor_docs:
        tokenized_doc = kor_tokenizer.tokenize(doc)
        print(tokenized_doc)

## Run
if __name__ == '__main__':
    docs = ['I am a boy!', 'He is a boy..', 'She is a girl?']

    test_normalizer()
    test_english_tokenizer()

    kor_docs = ['코퍼스의 첫 번째 문서입니다.', '두 번째 문서입니다.', '마지막 문서']

    print('\nKorean Tokenizer Test')
    print(kor_docs)
    test_korean_tokenizer(kor_docs)