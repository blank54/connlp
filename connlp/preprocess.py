#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import re
from math import exp
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
        
        else:
            return self.noun_extractor.train_extract(text)
