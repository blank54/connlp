#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(1) #Do not print INFO
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

import sys
sys.path.append(config_path)
from connlp.preprocess import EnglishTokenizer
from connlp.embedding import Vectorizer
from connlp.analysis import NER_LabeledSentence, NER_Labels, NER_Corpus, NER_Model
tokenizer = EnglishTokenizer()
vectorizer = Vectorizer()


## Named Entity Recognition
def w2v_embedding():
    global data_sents

    tokenized_sents = [tokenizer.tokenize(sent) for sent in data_sents.values()]
    w2v_model = vectorizer.word2vec(docs=tokenized_sents)

    word2vector = vectorizer.get_word_vectors(w2v_model)
    feature_size = w2v_model.vector_size
    return word2vector, feature_size

def develop_ner_corpus():
    global label_dict, data_sents, data_labels, max_sent_len

    docs = []
    for tag, sent in data_sents.items():
        words = [str(w) for w in tokenizer.tokenize(text=sent)]
        labels = data_labels[tag]
        docs.append(NER_LabeledSentence(tag=tag, words=words, labels=labels))

    ner_labels = NER_Labels(label_dict=label_dict)
    ner_corpus = NER_Corpus(docs=docs, ner_labels=ner_labels, max_sent_len=max_sent_len)
    return ner_corpus

def develop_ner_model():
    global parameters

    word2vector, feature_size = w2v_embedding()
    ner_corpus = develop_ner_corpus()
    ner_corpus.word_embedding(word2vector=word2vector, feature_size=feature_size)

    ner_model = NER_Model()
    ner_model.initialize(ner_corpus=ner_corpus, parameters=parameters)
    return ner_model

def train_ner_model():
    global ner_model

    ner_model.train(parameters=parameters)
    ner_model.evaluate()


## Run
if __name__ == '__main__':
    label_dict = {'NON': 0,
                  'ORG': 1,}

    data_sents = {'sent1': 'Sam works at Samsung',
                  'sent2': 'Sam likes to eat pizza',
                  'sent3': 'Peter and Sam are friends',
                  'sent4': 'Flora plays basketball',}
    data_labels = {'sent1': [1, 0, 0, 0],
                   'sent2': [1, 0, 0, 0, 0],
                   'sent3': [1, 0, 1, 0, 0],
                   'sent4': [1, 0, 0],}

    max_sent_len = 10

    parameters = {
        'lstm_units': 512,
        'lstm_return_sequences': True,
        'lstm_recurrent_dropout': 0.2,
        'dense_units': 100,
        'dense_activation': 'relu',
        'test_size': 0.3,
        'batch_size': 1,
        'epochs': 2,
        'validation_split': 0.1,
    }

    ner_model = develop_ner_model()
    train_ner_model()