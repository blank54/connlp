#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)

from connlp.util import makedir
from connlp.web_crawling import NewsQueryParser, NaverNewsListScraper, NaverNewsArticleParser
query_parser = NewsQueryParser()
list_scraper = NaverNewsListScraper()
article_parser = NaverNewsArticleParser()

import itertools
import pickle as pk
from tqdm import tqdm


## Parse query
def parse_query(fpath_query):
    query_list, date_list = query_parser.parse(fpath_query=fpath_query)
    return query_list, date_list

## Scrape URL list
def save_url_list(query, date, url_list):
    global fdir_url_list
    fname_url_list = 'Q-{}_D-{}.pk'.format(query, date)
    fpath_url_list = os.path.join(fdir_url_list, fname_url_list)
    makedir(fpath=fpath_url_list)
    with open(fpath_url_list, 'wb') as f:
        pk.dump(url_list, f)

def scrape_url_list(query_list, date_list):
    print('============================================================')
    print('URL list scraping')
    for date in sorted(date_list, reverse=False):
        print('  | Date: {}'.format(date))
        with tqdm(total=len(query_list)) as pbar:
            for query in query_list:
                url_list = list_scraper.get_url_list(query=query, date=date)
                save_url_list(query, date, url_list)
                pbar.update(1)

## Parse article
def load_url_list(fname_url_list):
    global fdir_url_list
    fpath_url_list = os.path.join(fdir_url_list, fname_url_list)
    with open(fpath_url_list, 'rb') as f:
        url_list = pk.load(f)
    return url_list

def save_article(article, fpath_article):
    makedir(fpath=fpath_article)
    with open(fpath_article, 'wb') as f:
        pk.dump(article, f)

def load_article(fpath_article):
    with open(fpath_article, 'rb') as f:
        article = pk.load(f)
    return article

def parse_article(fdir_url_list, fdir_article):
    total_num_urls = len(list(itertools.chain(*[load_url_list(fname) for fname in os.listdir(fdir_url_list)])))
    errors = []

    print('============================================================')
    print('Article parsing')
    with tqdm(total=total_num_urls) as pbar:
        for fname_url_list in os.listdir(fdir_url_list):
            query_list, _ = query_parser.urlname2query(fname_url_list=fname_url_list)
            url_list = load_url_list(fname_url_list=fname_url_list)

            for url in url_list:
                pbar.update(1)

                fname_article = 'a-{}.pk'.format(article_parser.url2id(url))
                fpath_article = os.path.join(fdir_article, fname_article)
                if not os.path.isfile(fpath_article):
                    try:
                        article = article_parser.parse(url=url)
                    except:
                        errors.append(url)
                        continue
                else:
                    article = load_article(fpath_article=fpath_article)

                article.extend_query(query_list)
                save_article(article=article, fpath_article=fpath_article)

    print('============================================================')
    print('  |Initial   : {:,} urls'.format(total_num_urls))
    print('  |Done      : {:,} articles'.format(len(os.listdir(fdir_article))))
    print('  |Error     : {:,}'.format(len(errors)))

    if errors:
        print('============================================================')
        print('Errors on articles:')
        for url in errors:
            print(url)


if __name__ == '__main__':
    ## Web crawling information
    fname_query = 'query_20210721.txt'
    fpath_query = os.path.join('test/web_crawling/naver/query/', fname_query)
    fdir_url_list = 'test/web_crawling/naver/url_list/'
    fdir_article = 'test/web_crawling/naver/article/'
    
    ## Parse query
    query_list, date_list = parse_query(fpath_query=fpath_query)

    ## Scrape URL list
    scrape_url_list(query_list=query_list, date_list=date_list)

    ## Parse article
    parse_article(fdir_url_list=fdir_url_list, fdir_article=fdir_article)