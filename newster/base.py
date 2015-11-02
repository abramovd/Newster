#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9

from __future__ import print_function
from algorithms.Ward import HierarchicalClustering
from algorithms.STC import SuffixTreeClustering
from algorithms.FCA import FCAClustering
from algorithms.KMeans import kMeansClustering
from scraper import Scraper, search_articles
from config import api_urls, api_keys

NUM_OF_CLUSTERS = 6

class Newster:
    def __init__(self, api_urls, api_keys, query = ''):
        self.API_URLs = api_urls
        self.API_KEYs = api_keys
        self.snippets = []
        self.sources = []
        self.links = []
        self.titles = []
        if len(query) > 0:
            result = search_articles(self.API_URLs, self.API_KEYs, query)
            self.snippets = result['snippets']
            self.sources = result['sources']
            self.links = result['links']
            self.titles = result['titles']
        self.clustering = None # stores clustering object
            
    def search(self, query):
        if len(query) > 0:
            self.snippets = search_articles(self.API_URLs, self.API_KEYs, query)
        return self.snippets
            
    def find_clusters(self, method, n_clusters = NUM_OF_CLUSTERS):
        if len(self.snippets) == 0:
            print("Sorry. There is nothing to cluster. Firstly, search for something.")
            return
        if method.lower() == "stc":
            self.clustering = SuffixTreeClustering(self.snippets)
        elif method.lower() == "fca":
            self.clustering = FCAClustering(self.snippets)
        elif method.lower() == "ward":
            self.clustering = HierarchicalClustering(self.snippets)
        elif method.lower() == "k-means" or method.lower() == "kmeans":
            self.clustering = kMeansClustering(self.snippets)
        else:
            print("Sorry, unknown clustering algorithm.")
            return {}
   
        self.clustering.find_clusters(n_clusters)
        
        return self.get_clusters()
        
    def get_snippets(self):
        return self.snippets
        
    def get_links(self):
        return self.links
        
    def get_sources(self):
        return self.sources
        
    def get_titles(self):
        return self.titles
        
    def print_snippets(self):
        for num, snippet in enumerate(self.snippets):
            print("Snippet #%i: " % num, end = ' ')
            print(snippet)
            print("--------------------------------")

    def print_links(self):
        for num, link in enumerate(self.links):
            print("URL for #%i: " % num, end = ' ')
            print(link)
            print("--------------------------------")
            
    def print_sources(self):
        for num, source in enumerate(self.sources):
            print("Source for #%i: " % num, end = ' ')
            print(source)
            print("--------------------------------")

    def print_titles(self):
        for num, title in enumerate(self.titles):
            print("Title for #%i: " % num, end = ' ')
            print(title)
            print("--------------------------------")
            
    def print_search_results(self):
         for item in range(len(self.snippets)):
             print("Search result #%i" % item)
             print("Title: ", end = '')
             print(self.titles[item])
             print("Snippet: ", end = '')
             print(self.snippets[item])
             print("URL: ", end = '')
             print(self.links[item])
             print("Source: ", end = '')
             print(self.sources[item])
             print('-------------------------------')
    
    def get_clusters(self):
        if self.clustering:
            return self.clustering.get_clusters()
        else:
            return {}
    
    def print_clusters(self):
        if self.clustering:
            return self.clustering.print_clusters()
    

def main():
    query = "Obama"
    newster = Newster(api_urls, api_keys, query)
    #newster.print_search_results()
    #newster.print_links()
    if len(newster.get_snippets()) > 0:
        print("--------------STC---------------")
        newster.find_clusters(method = "stc2", n_clusters = 6)
        newster.print_clusters()
        print("--------------FCA---------------")
        newster.find_clusters(method = "fca", n_clusters = 10)
        newster.print_clusters()
        print("-------------KMeans-------------")
        newster.find_clusters(method = "kmeans", n_clusters = 6)
        newster.print_clusters()
        print("--------------Ward---------------")
        newster.find_clusters(method = "ward", n_clusters = 6)
        newster.print_clusters()
    
if __name__ == "__main__":
    main()