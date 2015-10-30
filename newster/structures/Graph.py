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