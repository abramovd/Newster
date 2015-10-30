#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9

from __future__ import print_function
from algorithms.Ward import HierarchicalClustering
from algorithms.STC import SuffixTreeClustering
from algorithms.FCA import FCAClustering
from algorithms.KMeans import kMeansClustering
from Scraper import Scraper, search_articles
from config import api_urls, api_keys

NUM_OF_CLUSTERS = 6

class Newster:
    def __init__(self, api_urls, api_keys, query = ''):
        self.API_URLs = api_urls
        self.API_KEYs = api_keys
        self.snippets = []
        if len(query) > 0:
            self.snippets = search_articles(self.API_URLs, self.API_KEYs, query)
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
            self.clustering.find_final_clusters(n_clusters)
        elif method.lower() == "fca":
            self.clustering = FCAClustering(self.snippets)
            self.clustering.find_clusters(n_clusters)
        elif method.lower() == "ward":
            self.clustering = HierarchicalClustering(self.snippets)
            self.clustering.find_clusters(n_clusters)
        elif method.lower() == "k-means" or method.lower() == "kmeans":
            self.clustering = kMeansClustering(self.snippets)
            self.clustering.find_clusters(n_clusters)
        else:
            print("Sorry, unknown clustering algorithm.")
            return
        return self.get_clusters()
        
    def get_snippets(self):
        return self.snippets
        
    def print_snippets(self):
        for num, snippet in enumerate(self.snippets):
            print("Snippet #%i: " % num, end = ' ')
            print(snippet)
            print("--------------------------------")
    
    def get_clusters(self):
        return self.clustering.get_clusters()
    
    def print_clusters(self):
        return self.clustering.print_clusters()
    

def main():
    query = "Obama Kanye"
    newster = Newster(api_urls, api_keys, query)
    newster.print_snippets()
    if len(newster.get_snippets()) > 0:
        print("--------------STC---------------")
        newster.find_clusters(method = "stc", n_clusters = 6)
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