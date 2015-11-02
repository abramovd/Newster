#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9

from __future__ import division, print_function
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import operator

from structures.Graph import GraphNode, connected_components
from structures.SuffixTree import SuffixTree
from preprocessing.tokenize_and_stem import tokenize_and_stem
from scraper import Scraper, search_articles
from config import api_urls, api_keys

# global constants

ALPHA = 0.5 #for base-clusters' scores computing
BETA = 0.5 # penalty constant
K = 500 # max number of base clusters for merging
NUM_OF_FINAL_CLUSTERS = 7

class SuffixTreeClustering:
    """
    Class for suffix tree clustering
    """
    cluster_document = {} #base cluster -> documents it covers
    phrases = {} #phrases for each base cluster
    scores = {} #scores for base clusters
    sorted_clusters = [] #sorted base-clusters by the scores
    final_clusters = [] #final merged clusters
    top_final_clusters = [] #top n final clusters

    def __init__(self, snippets = []):
        """
        Args:
            snippets - list of strings where every element is a news snippet -
            not required. You can just use it without parametrs:
            STC = SuffixTreeClustering()
            STC.add_strings(snippet)
        """
        self.snippets = snippets
        self.suffix_tree = SuffixTree()
        if len(snippets) > 0:
            self.add_strings(snippets)
        
    def add_strings(self, strings):
        """
        strings - strings (snippets) to add to suffix tree 
        """
        for string in strings:
            if string is not None:
                self.suffix_tree.append_string(tokenize_and_stem(string))
        self.suffix_tree.fix_input_string()

    def find_base_clusters(self, node = None):
        """
        Find base clusters, recursive
        """
        if node is None:
            node = self.suffix_tree.root

        if(len(node.edges.values()) > 0):
            for edge in node.edges.keys():
                child = node.edges[edge]
                self.find_base_clusters(child)

                #if the child is a cluster - the parent is a cluster too
                if self.cluster_document.get(child.identifier) != None and (child.parent != self.suffix_tree.root):
                    #clusters.append(child.parent.identifier)
                    if self.phrases.get(child.parent.identifier) is None:
                        self.phrases[child.parent.identifier] = child.parent.phrase
                    #if child.parent.edge
                    if self.cluster_document.get(child.parent.identifier) == None:
                        self.cluster_document[child.parent.identifier] = self.cluster_document[child.identifier][:]
                    else:
                        self.cluster_document[child.parent.identifier] += self.cluster_document[child.identifier]

        else:
            if node.parent != self.suffix_tree.root:
                #clusters.append(node.parent.identifier)
                if self.phrases.get(node.parent.identifier) is None:
                    self.phrases[node.parent.identifier] = node.parent.phrase
                if self.cluster_document.get(node.parent.identifier) == None:
                    temp = []
                    temp.append(node.bit_vector)
                    self.cluster_document[node.parent.identifier] = temp[:]
                else:
                    self.cluster_document[node.parent.identifier].append(node.bit_vector)
        return
        
    def count_scores(self):
        """
        Count scores for base clusters
        Formula: Score(S) = |B| F(|P|),
        where |B| is the size of the cluster (number of covered documents),
        |P| is the number of words in the phrase
           """

        for cluster in self.phrases.keys():
            self.scores[cluster] = len(self.cluster_document[cluster])*F(len(self.phrases[cluster].split(' ')))
        return
    def similarity(self, base_clusters):
        """
        Compute Similarity Matrix
        Args:
            base_clusters - top (<= k = 500) sorted by score base clusters
        Return:
            Similarity Matrix of clusters
        """
        Sim = [[0 for x in range(len(base_clusters))] for x in range(len(base_clusters))]

        for i in range(len(base_clusters) - 1):
            Sim[i][i] = 1
            for j in range(i + 1, len(base_clusters)):
             B1 = self.cluster_document[base_clusters[i]]
             B2 = self.cluster_document[base_clusters[j]]
             intersec = set(B1).intersection(B2) # intersection of two clusters (common covered documents)
             if len(intersec) / len(B1) > ALPHA and len(intersec) / len(B2) > ALPHA:
                Sim[i][j] = 1
                Sim[j][i] = 1 #not important
        return Sim
    
    def merge_clusters(self, Sim):
        """
        Merging base clusters
        Args:
            Sim - matrix of similarity between base clusters
        """

        node_names = {} # dictionary ["name of base cluster"] = GraphNode
        for i in range(len(Sim)):
            if self.sorted_clusters[i] not in node_names.keys():
                node = GraphNode(self.sorted_clusters[i])
                node_names[self.sorted_clusters[i]] = node
            else:
                node = node_names[self.sorted_clusters[i]]
            for j in range(i + 1, len(Sim)): # efficency: checking only further clusters, ignoring previous
                if Sim[i][j] == 1:
                    if self.sorted_clusters[j] not in node_names.keys():
                        new_node = GraphNode(self.sorted_clusters[j])
                        node_names[self.sorted_clusters[j]] = new_node
                    node_names[self.sorted_clusters[i]].add_link(node_names[self.sorted_clusters[j]])
        number = 1
        for components in connected_components(node_names.values()):
            names = sorted(node.name for node in components)
            self.final_clusters.append(names)
            number += 1
    
    def find_clusters(self, number_of_clusters = NUM_OF_FINAL_CLUSTERS):
        """
        Findning final clusters
        Args:
            number_of_clusters - max number of final clusters
            default number of clusters = NUM_OF_FINAL_CLUSTERS = 10
        """
        if len(self.snippets) < number_of_clusters:
            print("Sorry, but number of snippets should be >= number of clusters")
            return {}
        
        self.sorted_clusters = []
        self.phrases = {}
        self.clusters_document = {}
        self.find_base_clusters()
        self.count_scores() # computing scores of each base claster
        # sorting base clusters by scores
        sorted_scores = sorted(self.scores.items(), key=operator.itemgetter(1), reverse=1)
        #print(len(sorted_scores))
        n = min(K, len(sorted_scores)) # number of selected top scored base clusters

        #selecting
        for i in range(n):
            self.sorted_clusters.append(sorted_scores[i][0])
        # computing Similarity matrix for selected clusters
        Sim = self.similarity(self.sorted_clusters)
        
        self.merge_clusters(Sim)
        # final clusters - result of merging

        # computing final scores for final clusters
        final_scores = {}

        for final_cluster_index in range(len(self.final_clusters)):
            sum = 0
            for base_cluster_index in range(len(self.final_clusters[final_cluster_index])):
                sum += self.scores[self.final_clusters[final_cluster_index][base_cluster_index]]
            final_scores[final_cluster_index] = sum

        sorted_final_scores = sorted(final_scores.items(), key=operator.itemgetter(1), reverse=1)

        # selecting top final clusters, the number of selecting is num_of_final_clusters = 10
     
        self.top_final_clusters = []
        n = min(number_of_clusters, len(self.final_clusters))
        for cluster in range(n):
            self.top_final_clusters.append(self.final_clusters[sorted_final_scores[cluster][0]])
            
        return self.get_clusters()
            
    def print_clusters(self):
        result = self.get_clusters()
        for cluster, snippets in result.items():
            print("cluster #%i contains documents: " % cluster, end = ' ')
            print(snippets)
            
    def get_clusters(self):
        result = {}
        count = 1
        for cluster in self.top_final_clusters:
            documents = []
            for base_cluster in cluster:
                documents.append(set(self.cluster_document[base_cluster]))
            result[count] = list(frozenset().union(*documents))
            count += 1
        
        return result
    
def F(P):
        """
        Penetializing function for computing score of a base cluster
        Needed for count_scores function
        Score(S) = |B| F(|P|)
        Args:
            P (here means |P|) - the length (number of words in the phrase of the base cluster)
        Return:
            float number - the result of function F for the cluster P
        """
        if P == 1:
            return 0
        elif P >= 2 and P <= 6:
            return P
        else:
            return BETA

def main():
        
    query = "obama"
    
    snippets = search_articles(api_urls, api_keys, query)['snippets']
    if len(snippets) == 0:
        print("Sorry, no results for your query!")
        return

    STC = SuffixTreeClustering(snippets)
    #STC.add_strings(snippets)
    #STC.find_base_clusters() # finding base clusters
    STC.find_clusters()
    STC.print_clusters()

if __name__ == "__main__":
    main()