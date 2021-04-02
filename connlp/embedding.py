#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(self, **kwargs):
        pass

    ## TODO
    # def tfidf(self, docs):
    #     vectorizer = TfidfVectorizer()
    #     tfidf_model = vectorizer.fit_transform(docs)
    #     return tfidf_model

    ## TODO: verbose
    def word2vec(self, docs, verbose=True, **kwargs):
        '''
        A method to develop a new Word2Vec model.
        Refer to: https://radimrehurek.com/gensim/models/word2vec.html

        Attributes
        ----------
        docs : list
            | a list of tokenized docs.
        parameters : dict
            | a dictionary of user-determined hyperparameters
        '''

        parameters = kwargs.get('parameters', {})
        model = Word2Vec(
            sentences=docs,
            size=parameters.get('size', 100),
            window=parameters.get('window', 5),
            min_count=parameters.get('min_count', 0),
            sg=parameters.get('sg', 0),
            hs=parameters.get('hs', 0),
            negative=parameters.get('negative', 5),
            ns_exponent=parameters.get('ns_exponent', 0.75),
            iter=parameters.get('iter', 5)
        )
        return model

    def word2vec_update(self, w2v_model, new_docs, verbose=True):
        '''
        A method to update an already developed Word2Vec model with new docs.
        Refer to: https://radimrehurek.com/gensim/models/word2vec.html

        Attributes
        ----------
        w2v_model : gensim.models.word2vec.Word2Vec
            | an already developed Word2Vec model.
        new_docs : list
            | a list of tokenized docs to update the Word2Vec model.
        '''

        w2v_model.min_count = 0
        w2v_model.build_vocab(sentences=new_docs, update=True)
        w2v_model.train(sentences=new_docs, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
        return w2v_model

    ## TODO: verbose
    def doc2vec(self, tagged_docs, verbose=True, **kwargs):
        '''
        A method to develop a new Doc2Vec model.
        REfer to: https://radimrehurek.com/gensim/models/doc2vec.html

        Attributes
        ----------
        tagged_docs : list
            | A list of tuples that include tag and tokenized sentence.
            | E.g.) [(tag1, [w1, w2, ...]), (tag2, [w3, w4, ...]), (tag3, [w5, ...])]
        parameters : dict
            | a dictionary of user-determined hyperparameters
        '''

        parameters = kwargs.get('parameters', {})
        docs_for_d2v = [TaggedDocument(words=doc, tags=[tag]) for tag, doc in tagged_docs]
        d2v_model = Doc2Vec(
            documents=docs_for_d2v,
            vector_size=parameters.get('vector_size', 100),
            window=parameters.get('window', 5),
            min_count=parameters.get('min_count', 0),
            dm=parameters.get('dm', 1),
            negative=parameters.get('negative', 5),
            epochs=parameters.get('epochs', 5),
            dbow_words=parameters.get('dbow_words', 1)
        )
        return d2v_model