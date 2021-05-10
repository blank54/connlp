#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)
from connlp.preprocess import EnglishTokenizer
from connlp.embedding import Vectorizer
tokenizer = EnglishTokenizer()
vectorizer = Vectorizer()


## Embedding
# TF-IDF
def test_tfidf_development():
    global docs

    tfidf_vectorizer, tfidf_matrix, tfidf_vocab = vectorizer.tfidf(docs=docs)
    print(type(tfidf_vectorizer))

    print(tfidf_matrix)
    print(tfidf_vocab)

# Word2Vec
def test_word2vec_development():
    global docs

    tokenized_docs = [tokenizer.tokenize(text=doc) for doc in docs]
    w2v_model = vectorizer.word2vec(docs=tokenized_docs)
    print(type(w2v_model))

    print(w2v_model.wv['boy'])
    print(w2v_model.wv.most_similar('boy', topn=3))

def test_word2vec_udpate():
    global docs, new_docs

    tokenized_docs = [tokenizer.tokenize(text=doc) for doc in docs]
    w2v_model = vectorizer.word2vec(docs=tokenized_docs)

    tokenized_new_docs = [tokenizer.tokenize(text=doc) for doc in new_docs]
    w2v_model_updated = vectorizer.word2vec_update(w2v_model=w2v_model, new_docs=tokenized_new_docs)
    print(type(w2v_model_updated))

    print(w2v_model_updated.wv['man'])
    print(w2v_model_updated.wv.most_similar('man', topn=3))

# Doc2Vec
def test_doc2vec_development():
    global docs, test_docs

    tagged_docs = [(idx, tokenizer.tokenize(text=doc)) for idx, doc in enumerate(docs)]
    d2v_model = vectorizer.doc2vec(tagged_docs=tagged_docs)
    print(type(d2v_model))

    print(d2v_model.infer_vector(doc_words=test_doc))


## Run
if __name__ == '__main__':
    docs = ['I am a boy', 'He is a boy', 'She is a girl']
    new_docs = ['Tom is a man', 'Sally is not a boy']
    test_doc = ['My', 'name', 'is', 'Peter']

    test_tfidf_development()
    # test_word2vec_development()
    # test_word2vec_udpate()
    # test_doc2vec_development()