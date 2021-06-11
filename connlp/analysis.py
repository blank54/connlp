#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

# Configuration
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import numpy as np
import pickle as pk
import itertools
from copy import deepcopy
from collections import defaultdict

import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from keras_contrib.layers import CRF
from keras.utils import to_categorical


class TopicModel:
    '''
    A class for topic modeling based on LDA.
    Refer to: https://radimrehurek.com/gensim/models/ldamodel.html

    Attributes
    ----------
    docs : dict
        | A dict of docs, of which key is tag and value is tokenized text.
    num_topics : int
        | The number of topics of the docs.

    Methods
    -------
    learn
        | Trains LDA model with given parameters.
        | Detail information for each parameter is provided in gensim website.
    assign
        | Assigns topic for each doc.
    '''

    def __init__(self, docs, num_topics):
        self.docs = docs
        self.id2word = corpora.Dictionary(self.docs.values())

        self.model = ''
        self.coherence = ''

        self.num_topics = num_topics
        self.docs_for_lda = [self.id2word.doc2bow(text) for text in self.docs.values()]
        
        self.tag2topic = defaultdict(int)
        self.topic2tag = defaultdict(list)

    def learn(self, **kwargs):
        parameters = kwargs.get('parameters', {})
        self.model = LdaModel(corpus=self.docs_for_lda,
                              id2word=self.id2word,
                              num_topics=self.num_topics,
                              iterations=parameters.get('iterations', 100),
                              update_every=parameters.get('update_every', 1),
                              chunksize=parameters.get('chunksize', 100),
                              passes=parameters.get('passes', 10),
                              alpha=parameters.get('alpha', 0.5),
                              eta=parameters.get('eta', 0.5),
                              )

        self.__calculate_coherence()

        # print('Learn LDA Model')
        # print('  | # of docs  : {:,}'.format(len(self.docs)))
        # print('  | # of topics: {:,}'.format(self.num_topics))

    def __calculate_coherence(self):
        coherence_model = CoherenceModel(model=self.model,
                                         texts=self.docs.values(),
                                         dictionary=self.id2word)
        self.coherence = coherence_model.get_coherence()

    def assign(self):
        doc2topic = self.model[self.docs_for_lda]
        
        for idx, tag in enumerate(self.docs):
            topic_id = sorted(doc2topic[idx], key=lambda x:x[1], reverse=True)[0][0]

            self.tag2topic[tag] = topic_id
            self.topic2tag[topic_id].append(tag)


class NER_Labels:
    '''
    A class that represents the NER labels.

    Attributes
    ----------
    label_dict : dict
        | A dict of NER labels of which keys are labels and values are index.
        | The label index should be start with 0.
    '''

    def __init__(self, label_dict):
        self.label_dict = label_dict
        self.label_list = ''
        self.label2id = ''
        self.id2label = ''

        self.__get_labels()

    def __get_labels(self):
        cnt = deepcopy(len(self.label_dict))
        self.label_dict['__PAD__'] = cnt
        self.label_dict['__UNK__'] = cnt+1

        self.label_list = list(self.label_dict.keys())
        self.label2id = self.label_dict
        self.id2label = {int(i): str(l) for i, l in enumerate(self.label_list)}

    def __len__(self):
        return len(self.label_list)

    def __iter__(self):
        for label in self.label_list:
            if label == '__PAD__' or label == '__UNK__':
                continue
            else:
                yield label


class NER_LabeledSentence:
    '''
    A class that represents the input of NER model.

    Attributes
    ----------
    tag : str
    words : list
    labels : NER_Labels
    '''

    def __init__(self, tag, words, labels):
        self.tag = tag
        self.words = words
        self.labels = labels

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return ' '.join(self.words)    


class NER_Corpus:
    '''
    A class that represents the NER corpus.

    Attributes
    ----------
    docs : list
        | A list of NER_LabeledSentence.
    ner_labels : NER_Label
        | An object of NER_Label.
    max_sent_len : int
        | The maximum length of sentences for analysis.

    Methods
    -------
    word_embedding
        | Converts each word to corresponding numeric vector based on Word2Vec.
    '''

    def __init__(self, docs, ner_labels, max_sent_len):
        self.docs = docs
        self.ner_labels = ner_labels
        self.max_sent_len = max_sent_len

        self.words = ''
        self.word2id = ''
        self.id2word = ''

        self.X_words_pad = ''
        self.Y_labels_pad = ''

        self.word2vector = ''
        self.feature_size = ''
        self.X_embedded = ''
        self.Y_embedded = ''

        self.__get_words()
        self.__sentence_padding()

    def __len__(self):
        return len(self.docs)

    def __get_words(self):
        self.words = list(set(itertools.chain(*[doc.words for doc in self.docs])))
        self.words.append('__PAD__')
        self.words.append('__UNK__')

        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.id2word = {i: w for i, w in enumerate(self.words)}

    def __sentence_padding(self):
        if not self.words:
            self.__get_words()
        else:
            pass

        X_words = []
        Y_labels = []
        for doc in self.docs:
            X_words.append([self.word2id[w] for w in doc.words])
            Y_labels.append(doc.labels)

        self.X_words_pad = pad_sequences(
            maxlen=self.max_sent_len,
            sequences=X_words,
            padding='post',
            value=self.word2id['__PAD__'])
        self.Y_labels_pad = pad_sequences(
            maxlen=self.max_sent_len,
            sequences=Y_labels,
            padding='post',
            value=self.ner_labels.label2id['__PAD__'])

    def word_embedding(self, word2vector, feature_size):
        '''
        Attributes
        ----------
        word2vector : dict
            | A dictionary of which keys are words and values are word vectors.
        feature_size : int
            | The number of dimension of the word vectors.
        '''

        self.word2vector = word2vector
        self.feature_size = feature_size
        self.word2vector['__PAD__'] = np.zeros(self.feature_size)
        self.word2vector['__UNK__'] = np.zeros(self.feature_size)

        X_embedded = np.zeros((self.__len__(), self.max_sent_len, self.feature_size))
        Y_embedded = np.zeros((self.__len__(), self.max_sent_len, len(self.ner_labels)))
        
        for i in range(len(self.docs)):
            for j, word_id in enumerate(self.X_words_pad[i]):
                Y_embedded[i] = to_categorical(self.Y_labels_pad[i], num_classes=(len(self.ner_labels)))
                for k in range(self.feature_size):
                    word = self.id2word[word_id]
                    X_embedded[i, j, k] = self.word2vector[word][k]

        self.X_embedded = X_embedded
        self.Y_embedded = Y_embedded


class NER_Dataset:
    '''
    A class that represents NER dataset.

    Attributes
    ----------
    X : numpy.array
        | Embedded sentences from NER corpus.
    Y : numpy.array
        | Embedded labels from NER corpus.
    test_size : float
        | The ratio of test size against the total.
    '''

    def __init__(self, X, Y, test_size):
        self.test_size = test_size
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=self.test_size)


class NER_Model:
    '''
    A class to conduct named entity recognition based on Bi-LSTM and CRF.
    
    If the model has not been trained yet, it needs to be initialized using "self.initialize()" method.
    Else if the model is already trained, it can be loaded using "self.load()" method.

    Methods
    -------
    initialize
        | Initialize the NER model with appropriate corpus and parameters.
    train
        | Train the NER model with appropriate parameters.
    evaluate
        | Show confusion matrix and F1 score of the NER model.
    predict
        | Predict the label of named entities from the given sentence.
    save
        | Save the NER model and the dataset.
    load
        | Initialize and load the NER model.
    '''

    def __init__(self):
        self.ner_labels = ''
        self.word2id = ''
        self.id2word = ''

        self.max_sent_len = ''
        self.feature_size = ''
        self.word2vector = ''
        
        self.lstm_units = ''
        self.lstm_return_sequences = ''
        self.lstm_recurrent_dropout = ''
        self.dense_units = ''
        self.dense_activation = ''

        self.X = ''
        self.Y = ''
        self.test_size = ''
        self.dataset = ''
        self.model = ''

        self.confusion_matrix = ''
        self.f1_score_list = ''
        self.f1_score_average = ''

    def initialize(self, ner_corpus, parameters):
        '''
        A method to initialize the NER model.

        Attributes
        ----------
        ner_corpus : NER_Corpus
            | Fully developed NER corpus.
        parameters : dict
            | Hyperparameters for Bi-LSTM layers.
        '''

        self.word2vector = ner_corpus.word2vector
        self.max_sent_len = ner_corpus.max_sent_len
        self.feature_size = ner_corpus.feature_size
        self.ner_labels = ner_corpus.ner_labels
        self.word2id = ner_corpus.word2id
        self.id2word = ner_corpus.id2word
        self.X = ner_corpus.X_embedded
        self.Y = ner_corpus.Y_embedded
        del ner_corpus

        self.lstm_units = parameters.get('lstm_units')
        self.lstm_return_sequences = parameters.get('lstm_return_sequences')
        self.lstm_recurrent_dropout = parameters.get('lstm_recurrent_dropout')
        self.dense_units = parameters.get('dense_units')
        self.dense_activation = parameters.get('dense_activation')

        _input = Input(shape=(self.max_sent_len, self.feature_size))
        model = Bidirectional(LSTM(units=self.lstm_units,
                                   return_sequences=self.lstm_return_sequences,
                                   recurrent_dropout=self.lstm_recurrent_dropout))(_input)
        model = TimeDistributed(Dense(units=self.dense_units,
                                      activation=self.dense_activation))(model)
        crf = CRF(len(self.ner_labels))
        _output = crf(model)

        model = Model(inputs=_input, outputs=_output)
        model.compile(optimizer='rmsprop',
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        
        self.model = model

    def train(self, parameters):
        '''
        A method to train the NER model.

        Attributes
        ----------
        parameters : dict
            | Hyperparameters for model training.
        '''

        self.test_size = parameters.get('test_size')
        self.batch_size = parameters.get('batch_size')
        self.epochs = parameters.get('epochs')
        self.validation_split = parameters.get('validation_split')

        self.dataset = NER_Dataset(X=self.X, Y=self.Y, test_size=self.test_size)
        self.model.fit(x=self.dataset.X_train,
                       y=self.dataset.Y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=self.validation_split,
                       verbose=True)

    def __pred2labels(self, sents, prediction):
        pred_labels = []
        for sent, pred in zip(sents, prediction):
            try:
                sent_len = np.where(sent==self.word2id['__PAD__'])[0][0]
            except:
                sent_len = self.max_sent_len
                
            labels = []
            for i in range(sent_len):
                labels.append(self.ner_labels.id2label[np.argmax(pred[i])])
            pred_labels.append(labels)
        return pred_labels

    def __get_confusion_matrix(self):
        matrix_size = len(self.ner_labels)-2
        matrix = np.zeros((matrix_size+1, matrix_size+1), dtype='int64')

        prediction = self.model.predict(self.dataset.X_test)
        pred_labels = self.__pred2labels(self.dataset.X_test, prediction)
        test_labels = self.__pred2labels(self.dataset.Y_test, self.dataset.Y_test)

        for i in range(len(pred_labels)):
            for j, pred in enumerate(pred_labels[i]):
                row = self.ner_labels.label2id[test_labels[i][j]]
                col = self.ner_labels.label2id[pred]
                matrix[row, col] += 1
                
        for i in range(matrix_size):
            matrix[i, matrix_size] = sum(matrix[i, 0:matrix_size])
            matrix[matrix_size, i] = sum(matrix[0:matrix_size, i])
            
        matrix[matrix_size, matrix_size] = sum(matrix[matrix_size, 0:matrix_size])
        self.confusion_matrix = matrix

    def __get_f1_score(self, p, r):
        if p != 0 or r != 0:
            return (2*p*r)/(p+r)
        else:
            return 0

    def __get_f1_score_from_matrix(self):
        f1_score_list = []
        matrix_size = len(self.confusion_matrix)
        for i in range(matrix_size):
            corr = self.confusion_matrix[i, i]
            pred = self.confusion_matrix[matrix_size-1, i]
            real = self.confusion_matrix[i, matrix_size-1]

            precision = corr/max(pred, 1)
            recall = corr/max(real, 1)
            f1_score_list.append(self.__get_f1_score(p=precision, r=recall))

        f1_score_average = np.mean(f1_score_list).round(3)
        self.f1_score_list = f1_score_list
        self.f1_score_average = f1_score_average


    def evaluate(self):
        '''
        A method to show model performance.
        '''

        self.__get_confusion_matrix()
        self.__get_f1_score_from_matrix()

        print('|--------------------------------------------------')
        print('|Confusion Matrix:')
        print(self.confusion_matrix)
        print('|--------------------------------------------------')
        print('|F1 Score: {:.03f}'.format(self.f1_score_average))
        print('|--------------------------------------------------')
        for category, f1_score in zip(self.ner_labels, self.f1_score_list):
            print('|    [{}]: {:.03f}'.format(category, f1_score))

    def predict(self, sent):
        '''
        A method to predict the label of named entities from the given sentence.

        Attributes
        ----------
        sent : list
            | A tokenized sentende.
        '''

        sent_by_id = []
        for w in [w.lower() for w in sent]:
            if w in self.word2id.keys():
                sent_by_id.append(self.word2id[w])
            else:
                sent_by_id.append(self.word2id['__UNK__'])

        sent_pad = pad_sequences(maxlen=self.max_sent_len, sequences=[sent_by_id], padding='post', value=self.word2id['__PAD__'])
        X_input = np.zeros((1, self.max_sent_len, self.feature_size), dtype=list)
        for j, w_id in enumerate(sent_pad[0]):
            for k in range(self.feature_size):
                word = self.id2word[w_id]
                X_input[0, j, k] = self.word2vector[word][k]

        prediction = self.model.predict(X_input)
        pred_labels = self.__pred2labels(sents=sent_pad, prediction=prediction)[0]
        return NER_Result(input_sent=sent, pred_labels=pred_labels)

    def save(self, fpath_model):
        '''
        A method to save the NER model and the dataset.

        fpath_model : str
            | Filepath of the model (.pk).
        '''

        self.model.save(fpath_model)
        fpath_dataset = '{}-dataset.pk'.format(fpath_model.replace('.pk', ''))
        with open(fpath_dataset, 'wb') as f:
            pk.dump(self.dataset, f)

    def load(self, fpath_model, ner_corpus, parameters):
        '''
        A method to initialize and load the NER model.

        Attributes
        ----------
        fpath_model : str
            | Filepath of the model (.pk).
        ner_corpus : NER_Corpus
            | Fully developed NER corpus.
        parameters : dict
            | A dictionary of parameters for Bi-LSTM layers.
        '''

        self.initialize(ner_corpus=ner_corpus, parameters=parameters)
        self.model.load_weights(fpath_model)
        fpath_dataset = '{}-dataset.pk'.format(fpath_model.replace('.pk', ''))
        with open(fpath_dataset, 'rb') as f:
            self.dataset = pk.load(f)


class NER_Result:
    '''
    A class that represents the NER prediction results

    Attributes
    ----------
    input_sent : list
        | A tokenized input sentence.
    pred_labels : list
        | A list of labels that correspond to the input_sent.
    '''

    def __init__(self, input_sent, pred_labels):
        self.sent = input_sent
        self.pred = pred_labels
        self.result = self.__assign_labels()

    def __assign_labels(self):
        result = defaultdict(list)
        for (word, label) in zip(self.sent, self.pred):
            result[label].append(word)
        return result

    def __iter__(self):
        for label in self.result:
            yield self.result[label]

    def __str__(self):
        output_sent = []
        for (word, label) in zip(self.sent, self.pred):
            output_sent.append('{}/{}'.format(word, label))
        return ' '.join(output_sent)