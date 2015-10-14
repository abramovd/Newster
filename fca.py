from __future__ import division, print_function
import json
import urllib2
import copy

class Context(object):

    def __init__(self, cross_table=[], objects=[], attributes=[]):
        """Create a context from cross table and list of objects, list
        of attributes

        cross_table - the list of bool lists
        objects - the list of objects
        attributes - the list of attributes
        """
        if len(cross_table) != len(objects):
            raise ValueError("Number of objects (=%i) and number of cross table"
                   " rows(=%i) must agree" % (len(objects), len(cross_table)))
        elif (len(cross_table) != 0) and len(cross_table[0]) != len(attributes):
            raise ValueError("Number of attributes (=%i) and number of cross table"
                    " columns (=%i) must agree" % (len(attributes),
                        len(cross_table[0])))

        self._table = cross_table
        self._objects = objects
        self._attributes = attributes

    def __deepcopy__(self, memo):
        return Context(copy.deepcopy(self._table, memo),
                       self._objects[:],
                       self._attributes[:])

    def get_objects(self):
        return self._objects

    objects = property(get_objects)

    def get_attributes(self):
        return self._attributes

    attributes = property(get_attributes)

    def examples(self):
        """Generator. Generate set of corresponding attributes
        for each row (object) of context
        """
        for obj in self._table:
            attrs_indexes = filter(lambda i: obj[i], range(len(obj)))
            yield set([self.attributes[i] for i in attrs_indexes])

    def intents(self):
        return self.examples()

    def get_object_intent_by_index(self, i):
        """Return a set of corresponding attributes for row with index i"""
        # TODO: !!! Very inefficient. Avoid using
        attrs_indexes = filter(lambda j: self._table[i][j],
                range(len(self._table[i])))
        return set([self.attributes[i] for i in attrs_indexes])

    def get_object_intent(self, o):
        index = self._objects.index(o)
        return self.get_object_intent_by_index(index)

    def get_attribute_extent_by_index(self, j):
        """Return a set of corresponding objects for column with index i"""
        objs_indexes = filter(lambda i: self._table[i][j],
                range(len(self._table)))
        return set([self.objects[i] for i in objs_indexes])

    def get_attribute_extent(self, a):
        index = self._attributes.index(a)
        return self.get_attribute_extent_by_index(index)

    def get_value(self, o, a):
        io = self.objects.index(o)
        ia = self.attributes.index(a)
        return self[io][ia]

    def add_attribute(self, col, attr_name):
        """Add new attribute to context with given name"""
        for i in range(len(self._objects)):
            self._table[i].append(col[i])
        self._attributes.append(attr_name)

    def add_column(self, col, attr_name):
        """Deprecated. Use add_attribute."""
        print("Deprecated. Use add_attribute.")
        self.add_attribute(col, attr_name)

    def add_object(self, row, obj_name):
        """Add new object to context with given name"""
        self._table.append(row)
        self._objects.append(obj_name)

    def add_object_with_intent(self, intent, obj_name):
        self._attr_imp_basis = None
        self._objects.append(obj_name)
        row = [(attr in intent) for attr in self._attributes]
        self._table.append(row)

    def add_attribute_with_extent(self, extent, attr_name):
        col = [(obj in extent) for obj in self._objects]
        self.add_attribute(col, attr_name)

    def set_attribute_extent(self, extent, name):
        attr_index = self._attributes.index(name)
        for i in range(len(self._objects)):
            self._table[i][attr_index] = (self._objects[i] in extent)

    def set_object_intent(self, intent, name):
        obj_index = self._objects.index(name)
        for i in range(len(self._attributes)):
            self._table[obj_index][i] = (self._attributes[i] in intent)

    def delete_object(self, obj_index):
        del self._table[obj_index]
        del self._objects[obj_index]

    def delete_object_by_name(self, obj_name):
        self.delete_object(self.objects.index(obj_name))

    def delete_attribute(self, attr_index):
        for i in range(len(self._objects)):
            del self._table[i][attr_index]
        del self._attributes[attr_index]

    def delete_attribute_by_name(self, attr_name):
        self.delete_attribute(self.attributes.index(attr_name))

    def rename_object(self, old_name, name):
        self._objects[self._objects.index(old_name)] = name

    def rename_attribute(self, old_name, name):
        self._attributes[self._attributes.index(old_name)] = name

    def transpose(self):
        """Return new context with transposed cross-table"""
        new_objects = self._attributes[:]
        new_attributes = self._objects[:]
        new_cross_table = []
        for j in xrange(len(self._attributes)):
            line = []
            for i in xrange(len(self._objects)):
                line.append(self._table[i][j])
            new_cross_table.append(line)
        return Context(new_cross_table, new_objects, new_attributes)

    def extract_subcontext_filtered_by_attributes(self, attributes_names,
                                                    mode="and"):
        """Create a subcontext with such objects that have given attributes"""
        values = dict( [(attribute, True) for attribute in attributes_names] )
        object_names, subtable = \
                            self._extract_subtable_by_attribute_values(values, mode)
        return Context(subtable,
                       object_names,
                       self.attributes)

    def extract_subcontext(self, attribute_names):
        """Create a subcontext with only indicated attributes"""
        return Context(self._extract_subtable(attribute_names),
                       self.objects,
                       attribute_names)

    def _extract_subtable(self, attribute_names):
        self._check_attribute_names(attribute_names)
        attribute_indices = [self.attributes.index(a) for a in attribute_names]
        table = []
        for i in range(len(self)):
            row = []
            for j in attribute_indices:
                row.append(self[i][j])
            table.append(row)

        return table

    def _extract_subtable_by_condition(self, condition):
        """Extract a subtable containing only rows that satisfy the condition.
        Return a list of object names and a subtable.

        Keyword arguments:
        condition(object_index) -- a function that takes an an object index and
            returns a Boolean value

        """
        indices = [i for i in range(len(self)) if condition(i)]
        return ([self.objects[i] for i in indices],
                [self._table[i] for i in indices])

    def _extract_subtable_by_attribute_values(self, values,
                                                    mode="and"):
        """Extract a subtable containing only rows with certain column values.
        Return a list of object names and a subtable.

        Keyword arguments:
        values -- an attribute-value dictionary

        """
        self._check_attribute_names(values.keys())
        if mode == "and":
            indices = [i for i in range(len(self)) if self._has_values(i, values)]
        elif mode == "or":
            indices = [i for i in range(len(self)) if self._has_at_least_one_value(i, values)]
        return ([self.objects[i] for i in indices],
                [self._table[i] for i in indices])

    def _has_values(self, i, values):
        """Test if ith object has attribute values as indicated.

        Keyword arguments:
        i -- an object index
        values -- an attribute-value dictionary

        """
        for a in values:
            j = self.attributes.index(a)
            v = values[a]
            if self[i][j] != v:
                return False
        return True

    def _has_at_least_one_value(self, i, values):
        """Test if ith object has at least one attribute value as in values.

        Keyword arguments:
        i -- an object index
        values -- an attribute-value dictionary

        """
        for a in values:
            j = self.attributes.index(a)
            v = values[a]
            if self[i][j] == v:
                return True
        return False

    def _check_attribute_names(self, attribute_names):
        if not set(attribute_names) <= set(self.attributes):
            wrong_attributes = ""
            for a in set(attribute_names) - set(self.attributes):
                wrong_attributes += "\t%s\n" % a
            raise ValueError("Wrong attribute names:\n%s" % wrong_attributes)

    ############################
    # Emulating container type #
    ############################

    def __len__(self):
        return len(self._table)

    def __getitem__(self, key):
        return self._table[key]

    ############################

    def __repr__(self):
        output = ", ".join(self.attributes) + "\n"
        output += ", ".join(self.objects) + "\n"
        cross = {True : "X", False : "."}
        for i in xrange(len(self.objects)):
            output += ("".join([cross[b] for b in self[i]])) + "\n"
        return output

class Concept(object):
    """
    A formal concept, contains intent and extent

    Examples
    ========
    Create a concept with extent=['Earth', 'Mars', 'Mercury', 'Venus']
    and intent=['Small size', 'Near to the sun'].

  #  >>> extent = ['Earth', 'Mars', 'Mercury', 'Venus']
  #  >>> intent = ['Small size', 'Near to the sun']
  #  >>> c = Concept(extent, intent)
  #  >>> 'Earth' in c.extent
  #  True
  #  >>> 'Pluto' in c.extent
  #  False
  #  >>> 'Small size' in c.intent
  #  True
  #  Print a concept.
  #  >>> print c
  #  (['Earth', 'Mars', 'Mercury', 'Venus'], ['Near to the sun', 'Small size'])
    """

    def __init__(self, extent, intent):
        """Initialize a concept with given extent and intent """
        self.extent = set(extent)
        self.intent = set(intent)
        self.meta = {}

    def __str__(self):
        """Return a string representation of a concept"""
        if len(self.intent) > 0:
            e = list(self.extent)
            e.sort()
        else:
            # TODO: Sometimes |intent| > 0, but extent is G.
            e = "G"
        if len(self.extent) > 0:
            i = list(self.intent)
            i.sort()
        else:
            # TODO: Sometimes |extent| > 0, but intent is M.
            i = "M"
        if len(self.meta.keys()) != 0:
            s = " meta: {0}".format(self.meta)
        else:
            s = ""
        return "({0}, {1}){2}".format(e, i, s)


def compute_covering_relation(cs):
        """Computes covering relation for a given concept system.
        Returns a dictionary containing sets of parents for each concept.
        Examples
        ========
        """
        parents = dict([(c, set()) for c in cs])

        for i in xrange(len(cs)):
            for j in xrange(len(cs)):
                if cs[i].intent < cs[j].intent:
                    parents[cs[j]].add(cs[i])
                    for k in xrange(len(cs)):
                        if cs[i].intent < cs[k].intent and\
                           cs[k].intent < cs[j].intent:
                                parents[cs[j]].remove(cs[i])
                                break
        return parents

def norris(context):

    # To be more efficient we store intent (as Python set) of every
    # object to the list
    # TODO: Move to Context class?
    examples = []
    for ex in context.examples():
        examples.append(ex)

    cs = [Concept([], context.attributes)]
    for i in xrange(len(context)):
        # TODO:
        cs_for_loop = cs[:]
        for c in cs_for_loop:
            if c.intent.issubset(examples[i]):
                c.extent.add(context.objects[i])
            else:
                new_intent = c.intent & examples[i]
                new = True
                for j in xrange(i):
                    if new_intent.issubset(examples[j]) and\
                       context.objects[j] not in c.extent:
                        new = False
                        break
                if new:
                    cs.append(Concept(set([context.objects[i]]) | c.extent,
                        new_intent))
    return (cs, compute_covering_relation(cs))

class ConceptLattice(object):
    """ConceptLattice class
    Examples
    ========

#    >>> from fca import (Context, Concept)
#    >>> ct = [[True, False, False, True],\
#              [True, False, True, False],\
#              [False, True, True, False],\
#              [False, True, True, True]]
#    >>> objs = ['1', '2', '3', '4']
#    >>> attrs = ['a', 'b', 'c', 'd']
#    >>> c = Context(ct, objs, attrs)
#    >>> cl = ConceptLattice(c)
#    >>> print cl
#    ([], M)
#    (['1'], ['a', 'd'])
#    (['2'], ['a', 'c'])
#    (['1', '2'], ['a'])
#    (['3', '4'], ['b', 'c'])
#    (['2', '3', '4'], ['c'])
#    (G, [])
#    (['4'], ['b', 'c', 'd'])
#    (['1', '4'], ['d'])
#    >>> print cl.parents(cl[5]) == set((cl[6],))
#    True
#    >>> print cl.children(cl[6]) == set((cl[5], cl[3], cl[8]))
#    True
    """
    def __init__(self, context, builder=norris):
        (self._concepts, self._parents) = norris(context)
        self._bottom_concept = [c for c in self._concepts if not self.ideal(c)][0]
        self._top_concept = [c for c in self._concepts if not self.filter(c)][0]
        self._context = context

    def get_context(self):
        return self._context

    context = property(get_context)

    def get_top_concept(self):
        # TODO: change
        return self._top_concept
    top_concept = property(get_top_concept)

    def get_bottom_concept(self):
        # TODO: change
        return self._bottom_concept

    bottom_concept = property(get_bottom_concept)

    def filter(self, concept):
        # TODO: optimize
        return [c for c in self._concepts if concept.intent > c.intent]

    def ideal(self, concept):
        # TODO: optimize
        return [c for c in self._concepts if c.intent > concept.intent]

    def __len__(self):
        return len(self._concepts)

    def __getitem__(self, key):
        return self._concepts[key]

    def __contains__(self, value):
        return value in self._concepts

    def __str__(self):
        s = ""
        for c in self._concepts:
            s = s + "%s\n" % str(c)
        return s[:-1]

    def index(self, concept):
        return self._concepts.index(concept)

    def parents(self, concept):
        return self._parents[concept]

    def children(self, concept):
        return set([c for c in self._concepts if concept in self.parents(c)])

from math import exp, log

def compute_probability(lattice):

    def get_intent_probability(B, p_m, n):
        ans = 0
        log_p_B = log_subset_probability(B, p_m)
        p_B = exp(log_p_B)
        if len(B) == 0:
            p_B = 1
            log_p_B = 0

        not_B = set()
        for attr in p_m.keys():
            if not attr in B:
                not_B.add(attr)
        for k in range(n + 1):
            mult = 0
            mult_is_zero = False
            for attr in not_B:
                try:
                    mult += log(1 - ((p_m[attr]) ** k))
                except:
                    mult_is_zero = True
                    break
            if mult_is_zero:
                continue
            try:
                if p_B == 1 and n == k:
                    return exp(mult)
                else:
                    t = k * log_p_B + (n - k) * log((1 - p_B)) + mult
                    t = exp(t)
                    # print k, t
            except:
                t = 0
            nom = range(n - k + 1, n + 1)
            den = range(1, k + 1)
            if len(den) != len(nom):
                print("False")
            for i in range(len(nom)):
                t *= nom[i] / float(den[i])
            ans += t
        return ans

    def log_subset_probability(subset, p_m):
        ans = 0
        for attr in subset:
            try:
                ans += log(p_m[attr])
            except:
                pass
        return ans

    context = lattice.context
    n = len(context)
    p_m = {}
    for attr in context.attributes:
        m_ = 0
        for i in range(n):
            o = context.get_object_intent_by_index(i)
            if attr in o:
                m_ += 1
        p_m[attr] = m_ / float(n)

    probability = {}
    for concept in lattice:
        probability[concept] = get_intent_probability(concept.intent, p_m, n)

    return probability

class ConceptSystem(object):
    """A ConceptSystem class contains a set of concepts
    Emulates container
    Examples
    ========

    >>> c = Concept([1, 2], ['a', 'b'])
    >>> cs = ConceptSystem([c])
    >>> c in cs
    True
    >>> Concept([1], ['c']) in cs
    False
    >>> print cs
    ([1, 2], ['a', 'b'])
    """
    def get_top_concept(self):
        # TODO: change
        return [c for c in self._concepts if not self.filter(c)][0]

    top_concept = property(get_top_concept)

    def get_bottom_concept(self):
        # TODO: change
        return [c for c in self._concepts if not self.ideal(c)][0]

    bottom_concept = property(get_bottom_concept)

    def filter(self, concept):
        # TODO: optimize
        return [c for c in self._concepts if concept.intent > c.intent]

    def ideal(self, concept):
        # TODO: optimize
        return [c for c in self._concepts if c.intent > concept.intent]

    def __init__(self, concepts=[]):
        self._concepts = concepts[:]
        self._parents = None
        self._bottom_concept = None
        self._top_concept = None

    def __len__(self):
        return len(self._concepts)

    def __getitem__(self, key):
        return self._concepts[key]

    def __contains__(self, value):
        return value in self._concepts

    def __str__(self):
        s = ""
        for c in self._concepts:
            if len(c.extent) > 1:
                s = s + "%s\n" % str(c)
        return s[:-1]

    def index(self, concept):
        return self._concepts.index(concept)

    def append(self, concept):
        if isinstance(concept, Concept):
            self._concepts.append(concept)
        else:
            raise TypeError("concept must be an instance of the Concept class")
        self._parents = None
        # TODO: optimize

    def remove(self, concept):
        if isinstance(concept, Concept):
            self._concepts.remove(concept)
        self._parents = None

    def compute_covering_relation(self):
        """Computes covering relation for a given concept system.
        Returns a dictionary containing sets of parents for each concept.
        Examples
        ========
        """
        cs = self
        parents = dict([(c, set()) for c in cs])

        for i in xrange(len(cs)):
            for j in xrange(len(cs)):
                if cs[i].intent < cs[j].intent:
                    parents[cs[j]].add(cs[i])
                    for k in xrange(len(cs)):
                        if cs[i].intent < cs[k].intent and\
                           cs[k].intent < cs[j].intent:
                                parents[cs[j]].remove(cs[i])
                                break
        return parents

    def parents(self, concept):
        if not self._parents:
            self._parents = self.compute_covering_relation()
        return self._parents[concept]

    def children(self, concept):
        return set([c for c in self._concepts if concept in self.parents(c)])

def compute_index(lattice, function, name):
    indexes = function(lattice)

    for concept in indexes.items():
        if concept[0].meta:
            concept[0].meta[name] = concept[1]
        else:
            concept[0].meta = {name : concept[1]}

def filter_concepts(lattice, function, mode, opt=1):
    """Return new concept system, filtered by function according to the mode.

    Modes:
    --- "part" - part of initial concept lattice
    --- "abs" - absolute value of the concepts in resulting concept system
    --- "value" - value of the index

    Additionaly add attribute, containing inforamtion about indexes, to the new lattice
    """
    def _filter_value(lattice, indexes, value):
        filtered_concepts = [item for item in indexes.items() if item[1]>=value]
        return ConceptSystem([c[0] for c in filtered_concepts])

    def _filter_abs(lattice, indexes, n):
        cmp_ = lambda x,y: cmp(x[1], y[1])
        sorted_indexes = sorted(indexes.items(), cmp_, reverse=False)
        filtered_concepts = sorted_indexes[:int(n)]

        return ConceptSystem([c[0] for c in filtered_concepts])

    def _filter_part(lattice, indexes, part):
        n = int(len(lattice) * part)
        cmp_ = lambda x,y: cmp(x[1], y[1])
        sorted_indexes = sorted(indexes.items(), cmp_, reverse=True)
        filtered_concepts = sorted_indexes[:n]

        values = sorted_indexes
        eps = values[n-2][1]-values[n-1][1]

        other_concepts = sorted_indexes[n:]
        for concept in other_concepts:
            if abs(concept[1] - values[n][1]) < eps:
                filtered_concepts.append(concept)

        return ConceptSystem([c[0] for c in filtered_concepts])

    indexes = function(lattice)
    if indexes:
        if mode == "part":
            ret = _filter_part(lattice, indexes, opt)
        elif mode == "abs":
            ret = _filter_abs(lattice, indexes, opt)
        elif mode == "value":
            ret = _filter_value(lattice, indexes, opt)
    return ret

import nltk
import numpy as np

import re
import argparse

stopwords = nltk.corpus.stopwords.words('english')
#for i in range(len(stopwords):
#	stopwrds[i] = stopwords[i].decode('utf-8')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
#print(stopwords)

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
    #print(tokens)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
            #print(filtered_tokens)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    #print(stems)
    #print("\n\n")
    return stems

if __name__ == "__main__":
    snippets = []
    
    guardURL = 'http://content.guardianapis.com/search?q='
    nytURL = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q='
    key_g = ' ' # insert your guardian api-key
    key_nyt = ' ' # #insert your nyt api-key
    
    query = "putin"
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
    
    #print("\n\n")
    
    for i in result_nyt:
    	#print(k, end = ".")
    	#print(i['snippet'])
        k += 1
        snippets.append(tokenize_and_stem(i['snippet'])) 
       
    
	attrib = {}
    count = 0
    attrs = []
    #labels = [0, 0, 0, 1, 2, 2, 0, 1, 0, 3, 1, 11]
    for sn in snippets:
        for i in range(len(sn)):
            if sn[i] not in attrib.values():
                attrib[count] = sn[i]
                #print(sn[i])
                attrs.append(sn[i])

                count += 1
    #print(attrib)

    context = [[False for i in range(max(attrib.keys())+1)] for j in range(len(snippets))]
    #print(context)
    count = 0

    for i in range(len(snippets)):
        for j in attrib.keys():
            if attrib[j] in snippets[i]:
                context[i][j] = True
    #print('context')
    #print(context)
    objs = [str(i) for i in range(0,len(snippets))]
    #objs = ['1', '2', '3', '4']
    #attrs = ['a', 'b', 'c', 'd']
    ct = [[True, False, False, True],\
          [True, False, True, False],\
          [False, True, True, False],\
          [False, True, True, True]]
    c = Context(context, objs, attrs)
    cl = ConceptLattice(c)
    #print('lattice')
    from sklearn.metrics import pairwise_distances
    #compute_index(cl, compute_probability, "Probability")
    clusters = []
    cs = filter_concepts(cl, compute_probability, "abs", 1000)
    print(cs)
    
    '''
    for i in cs._concepts:
    	if len(i.extent) > 1:		
    		print(list(i.extent))
        #for j in i.extent:
        #    print(j, end = ' ')
        #print('\n')
            #for c in i:
            #    print(i.extent)
    #print(cs.extent)
    '''