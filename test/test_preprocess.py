#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)

from connlp.preprocess import Normalizer, EnglishTokenizer, KoreanTokenizer, StopwordRemover
normalizer = Normalizer()
eng_tokenizer = EnglishTokenizer()
pretrained_kor_tokenizer = KoreanTokenizer()
unsupervised_kor_tokenizer = KoreanTokenizer(pre_trained=False)
stopword_remover = StopwordRemover()

## Preprocess
# Normalizer
def test_normalizer():
    global eng_docs
    
    for doc in eng_docs:
        normalized_doc = normalizer.normalize(text=doc)
        print(normalized_doc)

# English Tokenizer
def test_english_tokenizer():
    global eng_docs
    
    for doc in eng_docs:
        tokenized_doc = eng_tokenizer.tokenize(text=doc)
        print(tokenized_doc)

# Korean Tokenizer
def test_korean_tokenizer(kor_docs, kor_tokenizer):
    print('\nTRAINING Korean Tokenizer...')
    kor_tokenizer.train(kor_docs)

    print()

    for doc in kor_docs:
        tokenized_doc = kor_tokenizer.tokenize(doc)
        nouns = kor_tokenizer.extract_noun(doc)
        print('Tokens:', tokenized_doc)
        print('\tNouns:', nouns)

    print('\nKorean Tokenizer Test')
    print(kor_docs)

# Stopword Remover
def test_stopword_remover():
    global eng_docs

    tokenized_docs = []
    for doc in eng_docs:
        normalized_doc = normalizer.normalize(text=doc)
        tokenized_doc = eng_tokenizer.tokenize(text=normalized_doc)
        tokenized_docs.append(tokenized_doc)

    fpath_stoplist = os.path.join(os.getcwd(), os.path.dirname(__file__), 'thesaurus/stoplist.txt')
    stopword_remover.initiate(fpath_stoplist=fpath_stoplist)

    stopword_remover.count_freq_words(docs=tokenized_docs)
    
    stopword_removed_docs = []
    for doc in tokenized_docs:
        stopword_removed_docs.append(stopword_remover.remove(sent=doc))

    print(stopword_removed_docs)
    stopword_remover.check_removed_words(docs=tokenized_docs, stopword_removed_docs=stopword_removed_docs)


## Run
if __name__ == '__main__':
    eng_docs = ['I am a boy!', 'He is a boy..', 'She is a girl?']
    kor_docs = ['코퍼스의 첫 번째 문서입니다.', '두 번째 문서입니다.', '마지막 문서']

    test_normalizer()
    test_english_tokenizer()
    print('\ncorpus:', kor_docs)

    print('Testing a pre-trained KoreanTokenizer...')
    test_korean_tokenizer(kor_docs, pretrained_kor_tokenizer)

    print('Testing an unsupervised KoreanTokenizer...')
    test_korean_tokenizer(kor_docs, unsupervised_kor_tokenizer)
    
    test_stopword_remover()
