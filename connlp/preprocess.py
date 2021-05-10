#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import re


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