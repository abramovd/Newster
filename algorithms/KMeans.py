#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9


from __future__ import print_function
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
from preprocessing.tokenize_and_stem import tokenize_and_stem
from Scraper import Scraper, search_articles

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from config import api_urls, api_keys

num_clusters = 7


class kMeansClustering:
    """
    News clustering with KMeans method.
    """

    def __init__(self, snippets):
        """
        Args:
            snippets - list of strings
        """
        self.snippets = snippets
        self.clusters = []
    
    def find_clusters(self):
        """
        Finding clusters.
        Requires sklearn library.
        """
    
        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=0.999, max_features=200000,
                                 min_df=0.001, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,1))
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.snippets) #fit the vectorizer to synopses
        terms = tfidf_vectorizer.get_feature_names()
        matrix = tfidf_matrix.todense()

        km = KMeans(n_clusters=num_clusters)
        km.fit(tfidf_matrix)
        
        self.clusters = km.labels_.tolist()     
        return self.clusters
        
    def get_clusters(self):
        """
        Return:
            dict of elements-clusters.
                Keys: clusters
                Values: news in respective clusters  
        """
        result = {}
        for i, cluster in enumerate(self.clusters):
            if cluster + 1 not in result:
                result[cluster + 1] = [i]
            else:
                result[cluster + 1].append(i)
        return result    
    
    def print_clusters(self):
        result = self.get_clusters()
        for cluster, snippets in result.items():
            print("cluster #%i contains documents: " % cluster, end = ' ')
            print(snippets)

def main():

    query = "obama"
    
    snippets = search_articles(api_urls, api_keys, query)
    if len(snippets) == 0:
        return
    km = kMeansClustering(snippets)
    km.find_clusters()
    #km.print_clusters()
    
if __name__ == "__main__":
    main()