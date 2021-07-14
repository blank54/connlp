#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import itertools
from math import exp
from collections import Counter

from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor
from konlpy import tag


class Normalizer:
    '''
    A class used to normalize text with predetermined character replacement rules.

    Attributes
    ----------
    do_lower : bool
        | To lower the text or not. (default : True)
    do_marking : bool
        | To mark several tokens to predetermined entities or not. (default : False)
        | The predetermined entities include "LINK", "REF", and "NUM".

    Methods
    -------
    normalize
        | Normalizes the input text and returns a clean text.
    '''

    def __init__(self, do_lower=True, do_marking=False, **kwargs):
        self.do_lower = do_lower
        self.do_marking = do_marking

    def __remove_trash_char(self, text):
        text = text.replace('?', '')
        text = re.sub('[^ \'\?\./0-9a-zA-Zㄱ-힣\n]', '', text)

        # Remove debris from format conversion
        text = text.replace(' / ', '/')
        text = re.sub('\.+\.', ' ', text)
        text = text.replace('\\\\', '\\').replace('\\r\\n', '')
        text = text.replace('\n', '  ')
        text = re.sub('\. ', '  ', text)
        text = re.sub('\s+\s', ' ', text).strip()

        # Lower the text
        if self.do_lower:
            text = text.lower()
        else:
            pass
        
        # Remove endmarks
        if text.endswith('\.'):
            text = text[:-1]
        else:
            pass
            
        return text

    def __marking(self, text):
        for i, w in enumerate(sent):
            if re.match('www.', str(w)):
                sent[i] = 'LINK'
            elif re.search('\d+\d\.\d+', str(w)):
                sent[i] = 'REF'
            elif re.match('\d', str(w)):
                sent[i] = 'NUM'
            else:
                continue

        return sent

    def normalize(self, text):
        '''
        A method to normalize the input text.

        Attributes
        ----------
        text : str
            | An input text in str type.
        '''

        if self.do_marking:
            text = self.__marking(text)
        else:
            pass
            
        return self.__remove_trash_char(text)

    ## TODO: udpate character replacement rules
    # def update_char_list(self, input_char_list, verbose=False):
    #     cnt_before = len(self.remain_char_list)
    #     self.remain_char_list = list(set(self.remain_char_list.extend(input_char_list)))
    #     cnt_after = len(self.remain_char_list)

    #     if verbose:
    #         print('|Update remain_char_list: [{}] chars --> [{}] chars'.format(cnt_before, cnt_after))


class EnglishTokenizer:
    '''
    A class to tokenize an English sentence.

    Attributes
    ----------
    n : int
        | ngram size. (default : 1)

    Methods
    -------
    tokenize
        | tokenizes the input sentence.
    '''

    def __init__(self):
        pass

    def tokenize(self, text):
        '''
        A method to tokenize the input text.

        Attributes
        ----------
        text : str
            | An input text in str type.
        '''

        result = [w for w in re.split(' |  |\n', text) if w]
        return result


class KoreanTokenizer:
    '''
    A class to tokenize a Korean sentence.

    Attributes
    ----------
    pre_trained : bool
        | If True, one of pre-trained Korean analyzer, provided by KoNLPy, will be used (default : True)
        | If False, unsupervised KoreanTokenizer is initialized, based on soynlp L-Tokenizer. Argument 'anaylzer' is ignored.
    analyzer : str
        | Type of KoNLPy analyzer (default : Hannanum)
        | Available analyzers are: Hannanum, Kkma, Komoran, Mecab, Okt
        | Note: Mecab needs to be installed separately before being used.

    Methods
    -------
    train
        | Trains KoreanTokenizer on a corpus, only when 'pre_trained' argument is False.
    tokenize
        | Tokenizes the input sentence and returns its tokens.
    extract_noun
        | Extracts nouns from the input sentence.
    
    '''

    def __init__(self, pre_trained=True, analyzer='Hannanum'):
        self.pre_trained = pre_trained

        if analyzer == 'Hannanum':
            self.analyzer = tag.Hannanum()
        elif analyzer == 'Kkma':
            self.analyzer = tag.Kkma()
        elif analyzer == 'Komoran':
            self.analyzer = tag.Komoran()
        elif analyzer == 'Mecab':
            self.analyzer = tag.Mecab()
        elif analyzer == 'Okt':
            self.analyzer = tag.Okt()
        else:
            if pre_trained == False:
                pass
            else:
                print('Enter a valid KoNLPy analyzer name.\n\tavailable: Hannanum, Kkma, Komoran, Mecab, Okt')

        self.WordExtractor = WordExtractor(min_frequency=0)
        self.noun_extractor = LRNounExtractor(verbose=False)
        self.word_score = {}

    def train(self, text):
        '''
        A method to train the KoreanTokenizer on a corpus.
        If KoreanTokenizer.pre_trained == False, this method does nothing.

        Attributes
        ----------
        text : str
            | An input text in str type
        '''

        if self.pre_trained == True:
            print('A pre-trained KoreanTokenizer is being used. No need to train it.')
            return

        else:
            self.WordExtractor.train(text)
            self.words = self.WordExtractor.extract()

            def calculate_word_score(word, score):
                cohesion = score.cohesion_forward
                branching_entropy = score.right_branching_entropy
                
                word_score = cohesion * exp(branching_entropy)

                return word_score

            self.word_score = {word:calculate_word_score(word, score) for word, score in self.words.items()}

    def tokenize(self, text):
        '''
        A method to tokenize input text.

        Attriubutes
        -----------
        text : str
            | An input text to be tokenized

        Output
        ------
        tokens : list
            | List of tokens (in str) that consist of the input text

        '''

        if self.pre_trained == True:
            return self.analyzer.morphs(text)

        else:
            if not self.word_score:
                print('An unsupervised KoreanTokenizer should be trained first, before tokenizing.')
                return
            
            self.tokenizer = LTokenizer(scores=self.word_score)

            result = self.tokenizer.tokenize(text)

            return result

    def extract_noun(self, text):
        '''
        A method to extract nouns from input text

        Attributes
        ----------
        text : str
            | An input text from which nouns will be extracted

        Output
        ------
        nouns : list
            | List of noun tokens (in str) in the input text
        '''

        if self.pre_trained == True:
            return self.analyzer.nouns(text)


class StopwordRemover:
    '''
    A class to remove stopwords based on user-customized stopword list.

    Attributes
    ----------
    fpath_stoplist : str
        | The filepath of user-customized stopword list.    
    stoplist : list
        | The list of stopwords.

    Methods
    -------
    initiate
        | To assign fpath_stoplist.
    count_freq_words
        | A method to count frequently appeared words from documents.
        | The user can figure out the representatives of stopwords with this method.
    check_removed_words
        | A method to check the stopwords were successfully removed.
    load_stoplist
        | A method to load user-customized stopword list.
    write_stoplist
        | A method to write sorted and set stopword list.
    remove
        | A method to remove stopwords from a tokenized sent.
    '''

    def __init__(self):
        self.fpath_stoplist = ''
        self.stoplist = []

    def initiate(self, fpath_stoplist):
        '''
        To assign fpath_stoplist.

        Attributes
        ----------
        fpath_stoplist : str
            | The filepath of user-customized stopword list.
        '''

        self.fpath_stoplist = fpath_stoplist

    def count_freq_words(self, docs, verbose=True, **kwargs):
        '''
        A method to count frequently appeared words from documents.
        The user can figure out the representatives of stopwords with this method.

        Attributes
        ----------
        docs : list
            | A list of tokenized sentences.
        verbose : bool
            | Whether to show word counts if verbose is True. (default : True)
        topn : int
            | The number of words that would be printed.
        '''

        counter = Counter(itertools.chain(*docs))

        if verbose:
            topn = kwargs.get('topn', len(counter))
            print('========================================')
            print('Word counts')
            for idx, (word, cnt) in enumerate(sorted(counter.items(), key=lambda x:x[1], reverse=True)[:topn]):
                print('  | [{:>,}] {}: {}'.format((idx+1), word, cnt))

        return counter

    def check_removed_words(self, docs, stopword_removed_docs, **kwargs):
        '''
        A method to check the stopwords were successfully removed.

        Attributes
        ----------
        docs : list
            | A list of tokenized sentences (i.e., The original input documents).
        stopword_removed_docs : list
            | A list of stopword removed sentences.
        topn : int
            | The number of words that would be printed.
        '''

        word_counter_before = self.count_freq_words(docs, verbose=False)
        word_counter_after = self.count_freq_words(stopword_removed_docs, verbose=False)

        words_before = [w for w, c in sorted(word_counter_before.items(), key=lambda x:x[1], reverse=True)]
        words_after = [w for w, c in sorted(word_counter_after.items(), key=lambda x:x[1], reverse=True)]

        print('========================================')
        print('Check stopwords removed')
        topn = kwargs.get('topn', len(words_before))
        for idx, word in enumerate(words_before[:topn]):
            if word in words_after:
                print('  | [{:>,}] BEFORE: {} -> AFTER: {}({:,})'.format((idx+1), word, word, word_counter_after[word]))
            else:
                print('  | [{:>,}] BEFORE: {}({:,}) -> '.format((idx+1), word, word_counter_before[word]))

    def load_stoplist(self):
        '''
        A method to load user-customized stopword list.
        '''

        if os.path.isfile(self.fpath_stoplist):
            with open(self.fpath_stoplist, 'r', encoding='utf-8') as f:
                self.stoplist = list(set([w.strip() for w in f.read().strip().split('\n')]))
        else:
            pass

    def write_stoplist(self, verbose=False):
        '''
        A method to write sorted and set stopword list.

        Attributes
        ----------
        verbose : bool
            | Whether to show the overwriting status of stoplist. (default : False)
        '''

        if not self.stoplist:
            print('WARNING: No stoplist exists. The current fpath will be overwritten.')
        else:
            if os.path.isfile(self.fpath_stoplist) and verbose:
                print('INFO: The current fpath will be overwritten with sorted version.')
            with open(self.fpath_stoplist, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.stoplist))

    def remove(self, sent):
        '''
        A method to remove stopwords from a tokenized sent.

        Attributes
        ----------
        sent : list
            | A tokenized sentence.
        '''

        self.load_stoplist()
        self.write_stoplist()
        stopword_removed_sent = [w.strip() for w in sent if w not in self.stoplist]
        return stopword_removed_sent