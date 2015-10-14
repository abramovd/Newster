#!/usr/bin/env python
# -*- coding: utf-8

# Dmitry Abramov
# Python v. 2.7.9

from __future__ import print_function
from __future__ import division
import sys
import operator

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.lancaster import LancasterStemmer

import json
import urllib2
import re
import argparse

END_OF_STRING = sys.maxint

# global
cluster_document = {} #base cluster -> documents it covers
phrases = {} #phrases for each base cluster
scores = {} #scores for base clusters
alpha = 0.5 #for base-clusters' scores computing
beta = 0.5 # penalty constant
k = 500 # max number of base clusters for merging
sorted_clusters = [] #sorted base-clusters by the scores
final_clusters = [] #final merged clusters
num_of_final_clusters = 10


import nltk

# english stopwords
stopwords = nltk.corpus.stopwords.words('english')

# english stemmer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")



def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    w = 0
    while w < len(tokens):
    	#print(tokens[w])

        if tokens[w].lower() in stopwords:
            del tokens[w]
            w -= 1
        w += 1
        
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
            #print(filtered_tokens)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

class SuffixTreeNode:
    """
    Suffix tree node class + a tree edge that points to this node.
    """
    new_identifier = 0

    def __init__(self, start=0, end=END_OF_STRING):
        self.identifier = SuffixTreeNode.new_identifier
        SuffixTreeNode.new_identifier += 1

        # suffix link is required by Ukkonen's algorithm
        self.suffix_link = None

        # child edges/nodes, each dict key represents the first letter of an edge
        self.edges = {}

        self.phrase = ''

        # reference to parent
        self.parent = None

        # bit vector shows to which strings this node belongs
        self.bit_vector = 0

        # edge info: start index and end index
        self.start = start
        self.end = end

    def add_child(self, key, start, end):
        """
        Create a new child node

        Agrs:
            key: a char that will be used during active edge searching
            start, end: node's edge start and end indices

        Returns:
            created child node

        """
        child = SuffixTreeNode(start=start, end=end)
        child.parent = self
        self.edges[key] = child
        return child

    def add_exisiting_node_as_child(self, key, node):
        """
        Add an existing node as a child

        Args:
            key: a char that will be used during active edge searching
            node: a node that will be added as a child
        """
        node.parent = self
        self.edges[key] = node

    def get_edge_length(self, current_index):
        """
        Get length of an edge that points to this node

        Args:
            current_index: index of current processing symbol (usefull for leaf nodes that have "infinity" end index)
        """
        return min(self.end, current_index + 1) - self.start

    def __str__(self):
        return 'id=' + str(self.identifier)


class SuffixTree:
    """
    Generalized suffix tree
    """

    def __init__(self):
        # the root node
        self.root = SuffixTreeNode()

        # all strings are concatenaited together. Tree's nodes stores only indices
        self.input_string = []

        # number of strings stored by this tree
        self.strings_count = 0

        # list of tree leaves
        self.leaves = []

    def append_string(self, input_string):
        """
        Add new string to the suffix tree
        """
        start_index = len(self.input_string)
        current_string_index = self.strings_count

        # each sting should have a unique ending
        input_string += str(current_string_index)

        # gathering 'em all together
        self.input_string += input_string
        self.strings_count += 1

        # these 3 variables represents current "active point"
        active_node = self.root
        active_edge = 0
        active_length = 0

        # shows how many
        remainder = 0

        # new leaves appended to tree
        new_leaves = []

        # main circle
        for index in range(start_index, len(self.input_string)):
            previous_node = None
            remainder += 1
            while remainder > 0:
                if active_length == 0:
                    active_edge = index

                if self.input_string[active_edge] not in active_node.edges:
                    # no edge starting with current char, so creating a new leaf node
                    leaf_node = active_node.add_child(self.input_string[active_edge], index, END_OF_STRING)
                    #print(self.input_string[active_edge])

                    # a leaf node will always be leaf node belonging to only one string
                    # (because each string has different termination)
                    leaf_node.bit_vector = current_string_index
                    new_leaves.append(leaf_node)

                    # doing suffix link
                    if previous_node is not None:
                        previous_node.suffix_link = active_node
                    previous_node = active_node
                else:
                    #
                    # an active edge
                    next_node = active_node.edges[self.input_string[active_edge]]

                    # walking down through edges (if active_length is bigger than edge length)
                    next_edge_length = next_node.get_edge_length(index)
                    if active_length >= next_node.get_edge_length(index):
                        active_edge += next_edge_length
                        active_length -= next_edge_length
                        active_node = next_node
                        continue

                    # current edge already contains the suffix we need to insert.
                    # Increase the active_length and go forward
                    if self.input_string[next_node.start + active_length] == self.input_string[index]:
                        active_length += 1
                        if previous_node is not None:
                            previous_node.suffix_link = active_node
                        previous_node = active_node
                        break

                    # splitting edge
                    split_node = active_node.add_child(
                        self.input_string[active_edge],
                        next_node.start,
                        next_node.start + active_length
                    )
                    next_node.start += active_length
                    split_node.add_exisiting_node_as_child(self.input_string[next_node.start], next_node)
                    leaf_node = split_node.add_child(self.input_string[index], index, END_OF_STRING)
                    leaf_node.bit_vector = current_string_index
                    new_leaves.append(leaf_node)

                    # suffix link again
                    if previous_node is not None:
                        previous_node.suffix_link = split_node
                    previous_node = split_node

                remainder -= 1

                # follow suffix link (if exists) or go to root
                if active_node == self.root and active_length > 0:
                    active_length -= 1
                    active_edge = index - remainder + 1
                else:
                    active_node = active_node.suffix_link if active_node.suffix_link is not None else self.root

        # update leaves ends from "infinity" to actual string end
        for leaf in new_leaves:
            leaf.end = len(self.input_string)
        self.leaves.extend(new_leaves)

    def fix_input_string(self, node=None):
        """
        Fixing added number for input string and getting phrases for clusters
        """
        if node is None:
            node = self.root

        for edge, child in node.edges.items():
            #child = node.edges[edge]
            label = self.input_string[child.start:child.end]
            if (label[-1] >= '0' and label[-1]<='9' and len(label)>1):
                label.pop()
            node.edges[' '.join(label)] = node.edges.pop(edge)
            if child.parent == self.root:
                child.phrase = ' '.join(label)
            else:
                if len(edge) > 0:
                    child.phrase = child.parent.phrase + ' ' + ' '.join(label)
                else:
                    child.phrase = child.parent.phrase

            self.fix_input_string(child)

        return

    def to_graphviz(self, node=None, output=''):
        """
        Show the tree as graphviz string. For debugging purposes only
        Use after fixing input string
        """
        if node is None:
            node = self.root
            output = 'digraph G {edge [arrowsize=0.4,fontsize=10];'
        output +=\
            str(node.identifier) + '[label="' +\
            str(node.identifier) + '\\n' + '{0:b}'.format(node.bit_vector).zfill(self.strings_count) + '"'
        if node.bit_vector == 2 ** self.strings_count - 1:
            output += ',style="filled",fillcolor="red"'
        output += '];'
    #    if node.suffix_link is not None:
    #        output += str(node.identifier) + '->' + str(node.suffix_link.identifier) + '[style="dashed"];'
        #for c in node.edges.values():
    #   print(len(node.edges.values()))

        for edge, child in node.edges.items():
            child = node.edges[edge]
            label = self.input_string[child.start:child.end]
            output += str(node.identifier) + '->' + str(child.identifier) + '[label="' + ' '.join(label) + '"];'
            output = self.to_graphviz(child, output)

        if node == self.root:
            output += '}'

        return output


    def find_clusters(self, node = None):
        """
        Find base clusters, recursive
        """

        if node is None:
            node = self.root

        if(len(node.edges.values()) > 0):
            for edge in node.edges.keys():
                child = node.edges[edge]
                self.find_clusters(child)

                #if the child is a cluster - the parent is a cluster too
                if cluster_document.get(child.identifier) != None and (child.parent != self.root):
                    #clusters.append(child.parent.identifier)
                    if phrases.get(child.parent.identifier) is None:
                        phrases[child.parent.identifier] = child.parent.phrase
                    #if child.parent.edge
                    if cluster_document.get(child.parent.identifier) == None:
                        cluster_document[child.parent.identifier] = cluster_document[child.identifier][:]
                    else:
                        cluster_document[child.parent.identifier] += cluster_document[child.identifier]
                #if find(clusters(child.identifier)) is not None:

        else:
            if node.parent != self.root:
                #clusters.append(node.parent.identifier)
                if phrases.get(node.parent.identifier) is None:
                    phrases[node.parent.identifier] = node.parent.phrase
                if cluster_document.get(node.parent.identifier) == None:
                    temp = []
                    temp.append(node.bit_vector)
                    cluster_document[node.parent.identifier] = temp[:]
                else:
                    cluster_document[node.parent.identifier].append(node.bit_vector)
        return

    def __str__(self):
        return self.to_graphviz()

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
        return beta

def count_scores():
    """
    Count scores for base clusters
    Formula: Score(S) = |B| F(|P|),
    where |B| is the size of the cluster (number of covered documents),
    |P| is the number of words in the phrase
    """

    for cluster in phrases.keys():
        scores[cluster] = len(cluster_document[cluster])*F(len(phrases[cluster].split(' ')))
    return

def similarity(base_clusters):
    """
    Compute Similarity Matrix
    Args:
        base_clusters - top (<= k = 500)sorted by score base clusters
    Return:
        Similarity Matrix of clusters
    """
    Sim = [[0 for x in range(len(base_clusters))] for x in range(len(base_clusters))]

    for i in range(len(base_clusters) - 1):
        Sim[i][i] = 1
        for j in range(i + 1, len(base_clusters)):
         B1 = cluster_document[base_clusters[i]]
         B2 = cluster_document[base_clusters[j]]
         intersec = set(B1).intersection(B2) # intersection of two clusters (common covered documents)
         if len(intersec) / len(B1) > alpha and len(intersec) / len(B2) > alpha:
            Sim[i][j] = 1
            Sim[j][i] = 1 #not important
    return Sim

class GraphNode(object):
    """
    Graph's Node is a base cluster.
    """
    def __init__(self, name):

        # Name of the node
        self.__name  = name
        #links (edges)
        self.__links = set()

    def add_name(self, name):
        self.__name  = name

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)

def connected_components(nodes):
    """
    Compute connected components in the Graph of base clusters
    Args:
        nodes - list of nodes
    Return:
        list of connected components
        single is a component too
    """
    # List of connected components found. The order is random.
    result = []

    # Make a copy of the set, so we can modify it.
    nodes = set(nodes)

    # Iterate while we still have nodes to process.
    while nodes:

        # Get a random node and remove it from the global set.
        n = nodes.pop()

        # This set will contain the next group of nodes connected to each other.
        group = {n}

        # Build a queue with this node in it.
        queue = [n]

        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:

            # Consume the next item from the queue.
            n = queue.pop(0)

            # Fetch the neighbors.
            neighbors = n.links

            # Remove the neighbors we already visited.
            neighbors.difference_update(group)

            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)

            # Add them to the group of connected nodes.
            group.update(neighbors)

            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

            # Add the group to the list of groups.
        result.append(group)

        # Return the list of final clusters.
    return result

def merge_clusters(Sim):
    """
    Merging base clusters
    Args:
        Sim - matrix of similarity between base clusters
    """

    node_names = {} # dictionary ["name of base cluster"] = GraphNode
    for i in range(len(Sim)):
        if sorted_clusters[i] not in node_names.keys():
            node = GraphNode(sorted_clusters[i])
            node_names[sorted_clusters[i]] = node
        else:
            node = node_names[sorted_clusters[i]]
        for j in range(i + 1, len(Sim)): # efficency: checking only further clusters, ignoring previous
            if Sim[i][j] == 1:
                if sorted_clusters[j] not in node_names.keys():
                    new_node = GraphNode(sorted_clusters[j])
                    node_names[sorted_clusters[j]] = new_node
                node_names[sorted_clusters[i]].add_link(node_names[sorted_clusters[j]])
    number = 1
    for components in connected_components(node_names.values()):
        names = sorted(node.name for node in components)
        final_clusters.append(names)
        #print("Cluster #%i: %s" % (number, names))
        number += 1

def main():

    snippets = []
    labels = []
    
    '''
    with open("Ebola.txt", 'r') as f:
        s = 0
        for line in f.readlines():
            snippet = line.split(': ')[1].rstrip()
            #print(snippet)
            #snippets.append(cleaning(snippet))
            snippets.append(tokenize_and_stem(snippet))
            #print(cleaning(snippet))
    '''
    
    snippets = []
    
    guardURL = 'http://content.guardianapis.com/search?q='
    nytURL = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q='
    key_g = ' ' # insert your guardian api-key
    key_nyt = ' ' # #insert your nyt api-key
    
    query = "obama"
    #print(tokenize_and_stem("this is the real sheet mate!"))

    url_g = guardURL + query + key_g
    url_nyt = nytURL + query + key_nyt
    
    jstrs_g = urllib2.urlopen(url_g).read()
    jstrs_nyt = urllib2.urlopen(url_nyt).read()
    
    t_g = jstrs_g.strip('()')
    t_nyt = jstrs_nyt.strip('()')
    
    tss_g = json.loads(t_g)
    tss_nyt = json.loads(t_nyt)
    
    result_g = tss_g['response']['results']
    result_nyt = tss_nyt['response']['docs']
    
    k = 0
    
    for i in result_g:
    	#print(k, end = ".")
    	k += 1
    	#print(i['webTitle'])
        snippets.append(tokenize_and_stem(i['webTitle']))
    
    print("\n\n")
    
    for i in result_nyt:
    	#print(k, end = ".")
    	#print(i['snippet'])
        k += 1
        snippets.append(tokenize_and_stem(i['snippet'])) 

    #print(cleaning("maximum reasonable my suddenly honey"))
    suffix_tree = SuffixTree() # creating our suffix tree
    #snippets = ['cat ate cheese', 'mouse ate cheese too', 'cat ate mouse too']

    for snippet in snippets:
        suffix_tree.append_string(snippet)
    #print('suff tree yes')
    # simple tests
    #s0 = ['cat', 'ate', 'cheese']
    #s1 = ['mouse', 'ate', 'cheese', 'too']
    #s2 = ['cat', 'ate', 'mouse', 'too']
    #s3 = ['dog', 'ate', 'cat']
    #s4 = ['dog', 'love', 'becon']
    #s5 = ['becon', 'tasty']
    #s6 = ['cheese', 'tasty', 'too']
    #s7 = ['super', 'testy', 'becon', 'too']
    #s8 = ['hot', 'cat', 'rabbit', 'carrot']

    #adding to the suffix tree new strings (texts, snippets)

    #suffix_tree.append_string(s0)
    #suffix_tree.append_string(s1)
    #suffix_tree.append_string(s2)
    #suffix_tree.append_string(s8)
    #suffix_tree.append_string(s3)
    #suffix_tree.append_string(s4)
    #suffix_tree.append_string(s5)
    #suffix_tree.append_string(s6)
    #suffix_tree.append_string(s7)

    suffix_tree.fix_input_string()
    #print('fix string yes')

    #print(suffix_tree.to_graphviz()) # if you want to see the graphviz representation of the tree

    suffix_tree.find_clusters() # finding base clusters
    #print('base clusters yes')
    #for c in clusters:
    #    print(c) #not important
    #for c in cluster_document.values():
    #    for p in c:
    #        print(phrases[p])
            #base cluster - list of documents covered by it
    #    print('\n')

    #print(phrases) # phrases for each base cluster

    count_scores() # computing scores of each base claster
    # sorting base clusters by scores
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=1)
    #print(len(sorted_scores))
    n = min(k, len(sorted_scores)) # number of selected top scored base clusters

    #selecting
    for i in range(n):
        sorted_clusters.append(sorted_scores[i][0])
    #print('sorted base clus yes')
    # computing Similarity matrix for selected clusters
    Sim = similarity(sorted_clusters)
    #print('sim yes')
    # print(Sim)

    # merging base clusters as connected components in a graph
    merge_clusters(Sim)
    #print('merge yes')
    # final clusters - result of merging

    # print(final_clusters)

    # computing final scores for final clusters
    final_scores = {}

    for final_cluster_index in range(len(final_clusters)):
        sum = 0
        for base_cluster_index in range(len(final_clusters[final_cluster_index])):
            sum += scores[final_clusters[final_cluster_index][base_cluster_index]]
        final_scores[final_cluster_index] = sum

    sorted_final_scores = sorted(final_scores.items(), key=operator.itemgetter(1), reverse=1)

    # selecting top final clusters, the number of selecting is num_of_final_clusters = 10
    top_final_clusters = []
    n = min(num_of_final_clusters, len(final_clusters))
    for cluster in range(n):
       top_final_clusters.append(final_clusters[sorted_final_scores[cluster][0]])

    #labels = [0, 0, 0, 1, 2, 2, 0, 1, 0, 3, 1, 11]

    # printing the result
    # Clusters and documents that they cover
    #print('top fin clusters yes')
    #print(top_final_clusters)
    print(cluster_document)
    for c in cluster_document:
    	print(c)
    	
    count = 1
    for cluster in top_final_clusters:
        documents = []
        # print(len(cluster))
        for base_cluster in cluster:
            documents.append(set(cluster_document[base_cluster]))
            #print(phrases[base_cluster])
        result = frozenset().union(*documents)
        print("cluster #%i contains documents: %s" % (count, result))
        count += 1
'''
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(snippets)):
        for j in range(i + 1, len(snippets)):
            if j >= len(snippets): break
            for cl in documents:
                if i in cl and j in cl and labels[i] == labels[j]:
                    tp += 1
                elif i in cl and j in cl and labels[i] != labels[j]:
                    fp += 1
                elif i in cl and j not in cl and labels[i] == labels[j]:
                    fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(precision)
    print(recall)
    fm = 2 * (precision * recall) / (precision + recall)
    print(fm)
'''


if __name__ == "__main__":
    main()