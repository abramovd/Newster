#!/usr/bin/env python
# -*- coding: utf-8
from __future__ import division, print_function
import json
import urllib2

class Scraper:

    def __init__(self, URL, API_key):
        self.URL = URL
        self.API_key = '&api-key=' + API_key
        
    def search(self, query):
        query = 'q=' + query
        self.URL += query + self.API_key
        jstrs = urllib2.urlopen(self.URL).read().strip('()')
        self.json = json.loads(jstrs)['response']
        if self.URL.find('nytimes') != -1:
            self.json = self.json['docs']
            return self.json
        elif self.URL.find('guard') != -1:
            self.json = self.json['results']
            return self.json
            
    def fields(self):
        if self.URL.find('nytimes') != -1:
            if len(self.json) > 0:
                return self.json[0].keys()
        elif self.URL.find('guard') != -1:
            if len(self.json) > 0:
                return self.json[0].keys()
    
    def show_result_by_fields(self, fields):
        for i in range(len(self.json)):
                for field in fields:
                    if field in self.fields():
                        print(field, end = ': ')
                        print(self.json[i][field])
    
    def get_result_by_field(self, field):
        result = []
        for news in self.json:
            if field in self.fields():
                result.append(news[field])
        return result

def search_articles(URLs, keys, query):
    if len(keys) != len(URLs):
        print("Different number of URLs and API-keys")
        return

    scrappers = []
    result = []
    
    for i in range(len(URLs)):
        scrappers.append(Scraper(URLs[i], keys[i]))
        scrappers[i].search(query)
        if URLs[i].find('nytimes') != -1:
            c = 'snippet'
        elif URLs[i].find('guard') != -1:
            c = 'webTitle'
        else:
            print("Sorry. Unknown API.")
            #return []
        result += scrappers[i].get_result_by_field(c)
    
    if len(result) == 0:
        print("Sorry, no results for your query!")
    return result