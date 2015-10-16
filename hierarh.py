#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9

from __future__ import print_function
import numpy as np
from preprocessing.tokenize_and_stem import tokenize_and_stem
from Scraper import Scraper, search_articles

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from config import api_urls, api_keys

n_clusters = 7

class HierarchicalClustering:
    def __init__(self, snippets, num_clusters):
        self.snippets = snippets
        self.clusters = []
        self.num_clusters = num_clusters
    
    def find_clusters(self):
        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=0.999, max_features=200000,
                                 min_df=0.001, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,1))

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.snippets) #fit the vectorizer to synopses

        dist = 1 - cosine_similarity(tfidf_matrix)
    
        ward = AgglomerativeClustering(n_clusters=self.num_clusters, linkage='ward').fit(dist)
        self.clusters = ward.labels_
        return self.clusters
    
    def get_clusters(self):
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
    
    hc = HierarchicalClustering(snippets, n_clusters)
    hc.find_clusters()
    #hc.print_clusters()

if __name__ == "__main__":
    main()
