#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9

from __future__ import division, print_function
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import copy

from Scraper import Scraper, search_articles
from fca.context import Context
from fca.concept import Concept
from fca.concept_lattice import ConceptLattice, compute_probability
from fca.ConceptSystem import ConceptSystem
from preprocessing.tokenize_and_stem import tokenize_and_stem
#from config import api_urls, api_keys

MAX_NUM_OF_CLUSTERS = 10

class FCAClustering:
    """
        Clustering using Formal Concept Analysis and probability index. 
    """
    def __init__(self, rawsnippets):
        self.snippets = []
        self.attrs = []
        self.attrib = {}
        self.objs = []
        self.context = None
        self.lattice = None
        self.concept_system = None
        
        self.rawsnippets = rawsnippets
        count = 0
        for snippet in rawsnippets:
            self.snippets.append(tokenize_and_stem(snippet))
            for word in self.snippets[-1]:
                if word not in self.attrib.values():
                    self.attrib[count] = word
                    self.attrs.append(word)
                    count += 1
                        
    def build_context(self):
        context = [[False for i in range(max(self.attrib.keys())+1)] for j in range(len(self.snippets))]

        for i, snippet in enumerate(self.snippets):
            for j, word in self.attrib.items():
                if word in snippet:
                    context[i][j] = True
    
        self.objs = [str(i) for i in range(0,len(self.snippets))]
        self.context = Context(context, self.objs, self.attrs)
    
    def build_lattice(self):
        if self.context is not None:
            self.lattice = ConceptLattice(self.context)
            
    def find_probabilities(self, n_clusters = MAX_NUM_OF_CLUSTERS):
        self.concept_system = self.lattice.filter_concepts(compute_probability, "abs", len(self.snippets) + n_clusters)
    
    def find_clusters(self, n_clusters = MAX_NUM_OF_CLUSTERS):
        """
        Findning clusters
        """
        if len(self.snippets) < n_clusters:
            print("Sorry, but number of snippets should be >= number of clusters")
            return {}
        self.build_context()
        self.build_lattice()
        self.find_probabilities(n_clusters)
        return self.get_clusters()

    def get_clusters(self):
        result = {}
        cs = self.concept_system
        if cs == None:
            return {}
        count = 1
        for concept in cs._concepts:
            if len(concept.extent) > 1:
                result[count] = [int(elem) for elem in concept.extent]
                count += 1
        return result
        
    def get_common_phrases(self, num = 2):

        def restemming(word, num_snippets):
            for num_snippet in num_snippets:
                tokenized_snippet = tokenize_and_stem(self.rawsnippets[num_snippet], stem = 0)
                for sn in tokenized_snippet:
                    if sn.find(word) != -1:
                        return sn
            return ''
    
        phrases = {}
        cs = self.concept_system
        if cs == None:
            return {}
        count = 1
        for concept in cs._concepts:
            if len(concept.extent) > 1:
                extent = [int(x) for x in list(concept.extent)]
                intent = list(concept.intent)
                for i in range(len(intent)):
                    if count not in phrases:
                        phrases[count] = []
                    restem = restemming(intent[i], extent)
                    if restem != '':
                        if len(phrases[count]) < num:
                            phrases[count].append(restem)
                count += 1
        return phrases
    
    def print_common_phrases(self, num = 2):         
        
        result = self.get_common_phrases(num = num)
        for cluster, phrases in result.items():
            print("cluster #%i tags: " % cluster, end = ' ')
            print(phrases)
        
    def print_clusters(self):
        result = self.get_clusters()
        for cluster, snippets in result.items():
            print("cluster #%i contains documents: " % cluster, end = ' ')
            print(snippets)
        

def compute_index(lattice, function, name):
    """
        Computing probability index
    """
    indexes = function(lattice)

    for concept in indexes.items():
        if concept[0].meta:
            concept[0].meta[name] = concept[1]
        else:
            concept[0].meta = {name : concept[1]}

if __name__ == "__main__":

    query = "obama"
    
    
    snippets = search_articles(api_urls, api_keys, query)['snippets']
    FC = FCAClustering(snippets)
    FC.find_clusters()
    FC.print_clusters()
    FC.print_common_phrases()