#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import re


class EnglishTokenizer:
    def __init__(self, **kwargs):
        self.n = kwargs.get('n', 1) # ngram size

    def tokenize(self, text):
        result = [w for w in re.split(' |  |\n', text) if w]
        return result


class Normalizer:
    def __init__(self, **kwargs):
        self.do_marking = kwargs.get('do_marking', False)

    def __remove_trash_char(self, text):
        text = re.sub('[^ \'\?\./0-9a-zA-Zㄱ-힣\n]', '', text.lower())

        text = text.replace(' / ', '/')
        text = re.sub('\.+\.', ' ', text)
        text = text.replace('\\\\', '\\').replace('\\r\\n', '')

        text = text.replace('\n', '  ')
        text = re.sub('\. ', '  ', text)
        text = re.sub('\s+\s', ' ', text).strip()
        
        if text.endswith('\.'):
            text = text[:-1]
            
        return text

    def __marking(self, text):
        for i, w in enumerate(sent):
            if re.match('www.', str(w)):
                sent[i] = 'LINK'
            elif re.search('\d+\d\.\d+', str(w)):
                sent[i] = 'REF'
            elif re.match('\d', str(w)):
                sent[i] = 'NUM'
        return sent

    def normalize(self, text):
        if self.do_marking:
            text = self.__marking(text)
            
        return self.__remove_trash_char(text)