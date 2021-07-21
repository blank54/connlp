#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

# Configuration
import time
import itertools
import numpy as np
import pickle as pk
from collections import defaultdict
from datetime import datetime, timedelta

from urllib import request
from urllib.parse import quote
from bs4 import BeautifulSoup


class Status:
    '''
    A class to print status of web crawling.

    methods
    -------
    query_history
        | Print the history of queries.
        | The query files should start with "query_".
    urls
        | Print the status of url list.
    '''

    def query_history(self, fdir_queries):
        history = defaultdict(list)

        for fname in sorted(os.listdir(fdir_queries)):
            if not fname.startswith('query_'):
                continue

            collected_date = fname.replace('.txt', '').split('_')[1]
            fpath = os.path.join(fdir_queries, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                query_file = f.read().split('\n\n')
                date_list, query_list = parse_query(query_file)

            history['collected_date'].append(collected_date)
            history['date_start'].append(date_list[0])
            history['date_end'].append(date_list[-1])
            history['num_query'].append(len(query_list))
            history['query'].append(', '.join(query_list))

        print('============================================================')
        print('Status: Queries')
        print('  | fdir: {}'.format(fdir_queries))
        print('  | {:>13} {:>10} {:>10} {:>12} {:>15}'.format('CollectedDate', 'DateStart', 'DateEnd', 'NumOfQuery', 'Query'))
        history_df = pd.DataFrame(history)
        for i in range(len(history_df)):
            collected_date = history_df.iloc[i]['collected_date']
            date_start = history_df.iloc[i]['date_start']
            date_end = history_df.iloc[i]['date_end']
            num_query = history_df.iloc[i]['num_query']
            query = history_df.iloc[i]['query']
            print('  | {:>13} {:>10} {:>10} {:>12} {:>12}, ...'.format(collected_date, date_start, date_end, num_query, query[:10]))

    def urls(self, fdir_urls):
        urls = []
        for fname in os.listdir(fdir_urls):
            fpath_urls = os.path.join(fdir_urls, fname)
            with open(fpath_urls, 'rb') as f:
                urls.extend(pk.load(f))

        urls_distinct = list(set(urls))
        print('============================================================')
        print('Status: URLs')
        print('  | fdir: {}'.format(fdir_urls))
        print('  | Total # of urls: {:,}'.format(len(urls_distinct)))

    def articles(self, fdir_articles):
        flist = os.listdir(fdir_articles)

        print('============================================================')
        print('Status: Articles')
        print('  | fdir: {}'.format(fdir_articles))
        print('  | Total: {:,}'.format(len(flist)))