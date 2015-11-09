#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import division, print_function
import json
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2


class Scraper:
    """
        Class for scraping articles in JSON
    """
    def __init__(self, URL, API_key):
        self.URL = URL
        self.API_key = '&api-key=' + API_key
        
    def search(self, query):
        """
        Search news for this query through the web site which API given in URL
        Args:
            query - string
        """
        query = 'q=' + self.fix_query(query)
        self.URL += query + self.API_key
        
        try:
            jstrs = urllib2.urlopen(self.URL).read()
            jstrs = jstrs.decode('utf-8').strip('()')
        except urllib2.URLError:
            print("Sorry. Your URL or API-key is invalid")
            return urllib2.URLError
         
        self.json = json.loads(jstrs)['response']
        
        if self.URL.find('nytimes') != -1:
            self.json = self.json['docs']
            return self.json
        elif self.URL.find('guard') != -1:
            self.json = self.json['results']
            return self.json
            
    def fix_query(self, query):
        return '+'.join(query.split(' '))
            
    def fields(self):
        """
        Returns all the fields given in JSON
        Return:
            list of fields' names
        """
        if self.URL.find('nytimes') != -1:
            if len(self.json) > 0:
                return self.json[0].keys()
            else:
                print("Sorry, no result for your query on The NY Times")
                return []
        elif self.URL.find('guard') != -1:
            if len(self.json) > 0:
                return self.json[0].keys()
            else:
                print("Sorry, no result for your query on The Guardian")
                return []
    
    def show_result_by_fields(self, fields):
        """
        For every news prints only given fields from JSON.
        Works only in console.
        Args:
           List of strings where every element is a field in JSON
           or just one string to print only one field
        """
        if fields is list:
            for field in fields:
                if field not in self.fields():
                    print('Sorry, JSON does not have field ' + field + '.')
                    del(field)             
            for i in range(len(self.json)):
                    for field in fields:
                        if field in self.fields():
                            print(field, end = ': ')
                            print(self.json[i][field])
                    print('------------------------')
        else:
            for i in range(len(self.json)):
                print(field, end = ': ')
                print(self.json[i][field])
                print('------------------------')
        
    
    def get_result_by_field(self, field, search_articles = 0, source = ''):
        """
        Return result stored by given JSON field for every news
        Args:
           field - JSON field which you want to see for every news
        Return:
           list of strings (e.g., snippets for get_result_by_field("snippet"))
        """
        result = []
        if field in self.fields():
            for news in self.json:
                if search_articles:
                    if source == 'NYT' and news['snippet'] != None:
                        result.append(news[field])
                    elif source == 'GUARD' and news['webTitle'] != None:
                        result.append(news[field])
                elif news[field] != None:
                    result.append(news[field])
        else:
            print('Sorry, JSON does not have field ' + field)
        return result

def search_articles(URLs, keys, query):
    """
        Convenient way to search for articles' snippets or if snippets not given
        Web Titles.
        Now works for NY Times and The Guardian API.
        ARGS:
            - list of API URLS or just one API URL as a string
            - ist of API KEYS or just one API KEY as a string
            - query to search articles for
    """        
    snippets, sources, links, titles = [], [], [], []
    result = {'snippets': snippets, 'sources': sources, 'links': links, 'titles': titles}
    
    if type(URLs) is list and type(keys) is list: 
        if len(keys) != len(URLs):
            print("Different number of URLs and API-keys")
            return {'snippets': [], 'sources': [], 'links': [], 'titles': []}
            
        scrapers = []
        for i in range(len(URLs)):
            scrapers.append(Scraper(URLs[i], keys[i]))
            if scrapers[i].search(query) == urllib2.URLError:
                return {'snippets': [], 'sources': [], 'links': [], 'titles': []}
            if URLs[i].find('nytimes') != -1:
                c = 'snippet'
                src = 'NYT'
                result['links'] += scrapers[i].get_result_by_field('web_url',
                search_articles = 1, source = src)
                titles = scrapers[i].get_result_by_field('headline', search_articles = 1, source = src)
                result['titles'] += [title['main'] for title in titles]
                result['sources'] += ['NYT'] * len(scrapers[i].get_result_by_field('source',
                search_articles = 1, source = src))
            elif URLs[i].find('guard') != -1:
                c = 'webTitle'
                src = 'GUARD'
                result['links'] += scrapers[i].get_result_by_field('webUrl', 
                search_articles = 1, source = src)
                result['titles'] += scrapers[i].get_result_by_field(c, search_articles = 1,
                source = src)
                result['sources'] += ['GUARD'] * len(scrapers[i].get_result_by_field('webUrl', 
                search_articles = 1, source = src))
            else:
                print("Sorry. Unknown API.")
            result['snippets'] += scrapers[i].get_result_by_field(c, search_articles = 1,
            source = src)

    elif type(URLs) is str and type(keys) is str: 
        scraper = Scraper(URLs, keys)
        if scraper.search(query) != urllib2.URLError:
            if URLs[i].find('nytimes') != -1:
                src = 'NYT'
                c = 'snippet'
                result['links'] += scraper.get_result_by_field('web_url',
                search_articles = 1, source = src)
                titles = scraper.get_result_by_field('headline',
                search_articles = 1, source = src)
                result['titles'] += [title['main'] for title in titles]
                result['sources'] += ['NYT'] * len(scraper.get_result_by_field('source',
                search_articles = 1, source = src))
            elif URLs[i].find('guard') != -1:
                src = 'GUARD'
                c = 'webTitle'
                result['links'] += scraper.get_result_by_field('webUrl',
                search_articles = 1, source = src)
                result['titles'] += scraper.get_result_by_field(c, search_articles = 1,
                source = src)
                result['sources'] += ['GUARD'] * len(scraper.get_result_by_field('webUrl',
                search_articles = 1, source = src))
            else:
                print("Sorry. Unknown API.")
            result['snippets'] += scraper.get_result_by_field(c, search_articles = 1,
            source = src)
        else:
            return {'snippets': [], 'sources': [], 'links': [], 'titles': []}
    else:
        print("Sorry, bad input arguments")
        return {'snippets': [], 'sources': [], 'links': [], 'titles': []}

    if len(result['snippets']) == 0:
        print("Sorry, no results for your query!")
    return result