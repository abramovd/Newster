from LatticBuild import norris
from ConceptSystem import ConceptSystem

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
        
    def filter_concepts(self, function, mode, opt=1):
        """Return new concept system, filtered by function according to the mode.

        Modes:
        --- "part" - part of initial concept lattice
        --- "abs" - absolute value of the concepts in resulting concept system
        --- "value" - value of the index

        Additionaly add attribute, containing inforamtion about indexes, to the new lattice
        """
        def _filter_value(self, indexes, value):
            filtered_concepts = [item for item in indexes.items() if item[1]>=value]
            return ConceptSystem([c[0] for c in filtered_concepts])

        def _filter_abs(self, indexes, n):
            cmp_ = lambda x,y: cmp(x[1], y[1])
            sorted_indexes = sorted(indexes.items(), cmp_, reverse=False)
            filtered_concepts = sorted_indexes[:int(n)]

            return ConceptSystem([c[0] for c in filtered_concepts])

        def _filter_part(self, indexes, part):
            n = int(len(self) * part)
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

        indexes = function(self)
        if indexes:
            if mode == "part":
                ret = _filter_part(self, indexes, opt)
            elif mode == "abs":
                ret = _filter_abs(self, indexes, opt)
            elif mode == "value":
                ret = _filter_value(self, indexes, opt)
        return ret

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

    from math import exp, log
    
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
