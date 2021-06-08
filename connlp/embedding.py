#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    '''
    A class for text embedding.

    Methods
    -------
    tfidf
        | Develops a new TF-IDF model.
    word2vec
        | Develops a new Word2Vec model.
    word2vec_update
        | Updates the already trained Word2Vec model.
    doc2vec
        | Develops a new Doc2Vec model.
    '''

    def __init__(self):
        pass

    def tfidf(self, docs, do_lower=True, stopwords=None):
        '''
        A method to develop a new TF-IDF model.
        Refer to: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        
        Attributes
        ----------
        docs : list
            | A list of docs.
        do_lower : boolean
            | To lower the text or not. (default : True)
        stopwords : dict
            | User-defined stopwords. (default : None)
        '''

        tfidf_vectorizer = TfidfVectorizer(lowercase=do_lower,
                                           stop_words=stopwords,
                                           token_pattern=r'\b\w+\b')
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
        tfidf_vocab = tfidf_vectorizer.vocabulary_
        return tfidf_vectorizer, tfidf_matrix, tfidf_vocab

    def word2vec(self, docs, **kwargs):
        '''
        A method to develop a new Word2Vec model.
        Refer to: https://radimrehurek.com/gensim/models/word2vec.html

        Attributes
        ----------
        docs : list
            | A list of tokenized docs.
        parameters : dict (optional)
            | A dictionary of user-determined hyperparameters.
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

    def word2vec_update(self, w2v_model, new_docs, **kwargs):
        '''
        A method to update an already developed Word2Vec model with new docs.
        Refer to: https://radimrehurek.com/gensim/models/word2vec.html

        Attributes
        ----------
        w2v_model : gensim.models.word2vec.Word2Vec
            | An already developed Word2Vec model.
        new_docs : list
            | A list of tokenized docs to update the Word2Vec model.
        '''

        w2v_model.min_count = 0
        w2v_model.build_vocab(sentences=new_docs, update=True)
        w2v_model.train(sentences=new_docs, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
        return w2v_model

    def get_word_vectors(self, model):
        '''
        A method to extract word vectors from Word2Vec model in the format of dictionary.

        Attributes
        ----------
        model : gensim.models.word2vec.Word2Vec
            | A Word2Vec model.
        '''

        word_vectors = {w: model.wv[w] for w in model.wv.vocab.keys()}
        return word_vectors

    def doc2vec(self, tagged_docs, **kwargs):
        '''
        A method to develop a new Doc2Vec model.
        Refer to: https://radimrehurek.com/gensim/models/doc2vec.html

        Attributes
        ----------
        tagged_docs : list
            | A list of tuples that include tag and tokenized sentence.
            | E.g.) [(tag1, [w1, w2, ...]), (tag2, [w3, w4, ...]), (tag3, [w5, ...])]
        parameters : dict (optional)
            | A dictionary of user-determined hyperparameters.
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