try:
    import sys
    END_OF_STRING = sys.maxint
except:
    END_OF_STRING = sys.maxsize

class SuffixTreeNode:
    """
    Suffix tree node class + a tree edge that points to this node.
    """
    new_identifier = 0

    def __init__(self, start=0, end = END_OF_STRING):
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

        for edge, child in node.edges.items():
            child = node.edges[edge]
            label = self.input_string[child.start:child.end]
            output += str(node.identifier) + '->' + str(child.identifier) + '[label="' + ' '.join(label) + '"];'
            output = self.to_graphviz(child, output)

        if node == self.root:
            output += '}'

        return output

    def __str__(self):
        return self.to_graphviz()