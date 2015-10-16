from fca.concept import Concept
from fca.ConceptSystem import ConceptSystem

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
    cs = ConceptSystem(cs)
    return (cs, cs.compute_covering_relation())