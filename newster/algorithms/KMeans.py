#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9


from __future__ import print_function
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
from preprocessing.tokenize_and_stem import tokenize_and_stem
from scraper import Scraper, search_articles

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#from config import api_urls, api_keys

NUM_OF_CLUSTERS = 7

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
    
    def find_clusters(self, n_clusters = NUM_OF_CLUSTERS):
        """
        Finding clusters.
        Requires sklearn library.
        """
        if len(self.snippets) < n_clusters:
            print("Sorry, but number of snippets should be >= number of clusters")
            return {}
    
        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=0.999, max_features=200000,
                                 min_df=0.001, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,1))
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.snippets) #fit the vectorizer to synopses
        terms = tfidf_vectorizer.get_feature_names()
        matrix = tfidf_matrix.todense()

        km = KMeans(n_clusters = n_clusters)
        km.fit(tfidf_matrix)
        
        self.clusters = km.labels_.tolist()
        self.order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        self.terms = tfidf_vectorizer.get_feature_names()
        
        return self.get_clusters()

    def get_common_phrases(self, num = 2):
        def restemming(word, num_snippets):
            for num_snippet in num_snippets:
                tokenized_snippet = tokenize_and_stem(self.snippets[num_snippet], stem = 0)
                for sn in tokenized_snippet:
                    if sn.find(word) != -1:
                        return sn
            return ''       

        phrases = {}
        for i in range(len(self.get_clusters().keys())):
            for ind in self.order_centroids[i, :num]:
                if i + 1 not in phrases:
                    phrases[i + 1] = []
                restem = restemming(self.terms[ind], self.get_clusters()[i + 1])
                if restem != '':
                    if len(phrases[i + 1]) < num:
                        phrases[i + 1].append(restem)
        return phrases

    def print_common_phrases(self, num = 2):         
        
        result = self.get_common_phrases(num = num)
        for cluster, phrases in result.items():
            print("cluster #%i tags: " % cluster, end = ' ')
            print(phrases)
   
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
    
    snippets = search_articles(api_urls, api_keys, query)['snippets']
    if len(snippets) == 0:
        return

    km = kMeansClustering(snippets)
    km.find_clusters()
    km.print_clusters()
    km.print_common_phrases()
    
if __name__ == "__main__":
    main()
