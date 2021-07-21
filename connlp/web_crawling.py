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


class NewsStatus:
    '''
    A class to print status of web crawling.

    Methods
    -------
    queries
        | Print the history of queries.
        | The query files should start with "query_".
    urls
        | Print the number of url list.
    articles
        | Print the number of article list.
    '''

    def queries(self, fdir_queries):
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


class NewsArticle:
    '''
    A class of news article.

    Attributes
    ----------
    url : str
        | The article url.
    id : str
        | The identification code for the article.
    query : list
        | A list of queries that were used to search the article.
    title : str
        | The title of the article.
    date : str
        | The uploaded date of the article. (format : yyyymmdd)
    category : str
        | The category that the article belongs to.
    content : str
        | The article content.
    content_normalized : str
        | Normalized content of the article.

    Methods
    -------
    extend_query
        | Extend the query list with the additional queries that were used to search the article.
    '''

    def __init__(self, **kwargs):
        self.url = kwargs.get('url', '')
        self.id = kwargs.get('id', '')
        self.query = []

        self.title = kwargs.get('title', '')
        self.date = kwargs.get('date', '')
        self.category = kwargs.get('category', '')
        self.content = kwargs.get('content', '')

        self.content_normalized = kwargs.get('content_normalized', '')

    def extend_query(self, query_list):
        '''
        A method to extend the query list.

        Attributes
        ----------
        query_list : list
            | A list of queries to be extended.
        '''

        queries = self.query
        queries.extend(query_list)
        self.query = list(set(queries))


class NewsQueryParser:
    '''
    A class of news query parser.

    Method
    ------
    return_query_list
        | Parse the query file and return the list of queries.
    return_date_list
        | Parse the query file and return the list of dates.
    parse
        | Parse the query file and return the list of queries and dates.
    '''

    def return_query_list(self, query_file):
        _splitted_queries = [queries.split('\n') for queries in query_file[1:]]
        _queries_combs = list(itertools.product(*_splitted_queries))
        query_list = ['+'.join(e) for e in _queries_combs]
        return query_list

    def return_date_list(self, query_file):
        date_start, date_end = query_file[0].split('\n')

        date_start_formatted = datetime.strptime(date_start, '%Y%m%d')
        date_end_formatted = datetime.strptime(date_end, '%Y%m%d')
        delta = date_end_formatted - date_start_formatted

        date_list = []
        for i in range(delta.days+1):
            day = date_start_formatted + timedelta(days=i)
            date_list.append(datetime.strftime(day, '%Y%m%d'))
        return date_list

    def parse(self, fpath_query):
        with open(fpath_query, 'r', encoding='utf-8') as f:
            query_file = f.read().split('\n\n')

        query_list = self.return_query_list(query_file=query_file)
        date_list = self.return_date_list(query_file=query_file)
        return query_list, date_list

    def urlname2query(self, fname_url_list):
        Q, D = fname_url_list.replace('.pk', '').split('_')
        query_list = Q.split('-')[1].split('+')
        date = D.split('-')[1]
        return query_list, date


class NewsQuery:
    '''
    A class of news query to address the encoding issues.

    Attributes
    ----------
    query : str
        | Query in string format.
    '''

    def __init__(self, query):
        self.query = query

    def __call__(self):
        return quote(self.query.encode('utf-8'))

    def __str__(self):
        return '{}'.format(self.query)

    def __len__(self):
        return len(self.query.split('+'))


class NewsDate:
    '''
    A class of news dates to address the encoding issues.

    Attributes
    ----------
    date : str
        | Date in string format. (format : yyyymmdd)
    formatted : datetime
        | Formatted date.
    '''

    def __init__(self, date):
        self.date = date
        self.formatted = self.__convert_date()

    def __call__(self):
        return self.formatted

    def __str__(self):
        return '{}'.format(self.__call__())

    def __convert_date(self):
        try:
            return datetime.strptime(self.date, '%Y%m%d').strftime('%Y.%m.%d')
        except:
            return ''


class NewsCrawler():
    '''
    A class of news crawler that includes headers.

    Attributes
    ----------
    time_lag_random : float
        | A random number for time lag.
    headers : dict
        | Crawling header that is used generally.
    '''

    time_lag_random = np.random.normal(loc=1.0, scale=0.1)
    headers = {'User-Agent': '''
        [Windows64,Win64][Chrome,58.0.3029.110][KOS] 
        Mozilla/5.0 Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/58.0.3029.110 Safari/537.36
        '''}


class NaverNewsListScraper(NewsCrawler):
    '''
    A scraper for the article list of naver news.

    Attributes
    ----------
    url_base : str
        | The url base of the article list page of naver news.

    Methods
    -------
    get_url_list
        | Get the url list of articles for the given query and dates.
    '''

    def __init__(self):
        self.url_base = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=1&photo=0&field=0&pd=3&ds={}&de={}&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:from{}to{},a:all&start={}'

    def get_url_list(self, query, date):
        '''
        A method to get url list of articles for the given query and dates.

        Attributes
        ----------
        query : str
            | A query of simple string format.
        date : str
            | A date to search the query. (foramt : yyyymmdd)
        '''

        query = NewsQuery(query)
        date = NewsDate(date)

        url_list = []
        start_idx = 1
        while True:
            url_list_page = self.url_base.format(query(), date(), date(), date.date, date.date, start_idx)
            req = request.Request(url=url_list_page, headers=self.headers)
            html = request.urlopen(req).read()
            soup = BeautifulSoup(html, 'lxml')
            time.sleep(self.time_lag_random)

            url_list.extend([s.get('href') for s in soup.find_all('a', class_='info') if '네이버뉴스' in s])
            start_idx += 10

            if soup.find('div', class_='not_found02'):
                break
            else:
                continue

        return list(set(url_list))


class NaverNewsArticleParser(NewsCrawler):
    '''
    A parser of naver news article page.

    Methods
    -------
    parse
        | Parse the page of the given url and return the article information.
    '''

    def __init__(self):
        pass

    def parse(self, url):
        '''
        A method to parse the article page.

        Attributes
        ----------
        url : str
            | The url of the article page.
        '''

        req = request.Request(url=url, headers=self.headers)
        html = request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag_random)

        title = soup.find_all('h3', {'id': 'articleTitle'})[0].get_text().strip()
        date = soup.find_all('span', {'class': 't11'})[0].get_text().split()[0].replace('.', '').strip()
        content = soup.find_all('div', {'id': 'articleBodyContents'})[0].get_text().strip()

        try:
            category = soup.find_all('em', {'class': 'guide_categorization_item'})[0].get_text().strip()
        except IndexError:
            category = None

        article = NewsArticle(url=url, id=self.url2id(url), title=title, date=date, category=category, content=content)
        return article

    def url2id(self, url):
        id = str(url.split('=')[-1])
        return id