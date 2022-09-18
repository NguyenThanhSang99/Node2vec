import numpy as np
import networkx as nx
import random

class Graph():
    def __init__(self, graph, is_directed, p, q):
        self.graph = graph
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def walk(self, length, started_node):

        graph = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walks = [started_node]
        while len(walks) < length:
            cur = walks[-1]
            cur_nbrs = sorted(graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walks) == 1:
                    walks.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev_node = walks[-2]
                    next_node = cur_nbrs[alias_draw(alias_edges[(prev_node, cur)][0], alias_edges[(prev_node, cur)][1])]
                    walks.append(next_node)
            else:
                break
        return walks


    # Get the alias edge
    def get_alias_edge(self, src, dst):
        graph = self.graph
        p = self.p
        q = self.q

        unnormalized_prods = []

        for dst_nbr in sorted(graph.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_prods.append(graph[dst][dst_nbr]['weight']/p)
            elif graph.has_edge(dst_nbr, src):
                unnormalized_prods.append(graph[dst][dst_nbr]['weight'])
            else:
                unnormalized_prods.append(graph[dst][dst_nbr]['weight']/q)

            norm_const = sum(unnormalized_prods)
            normalized_prods = [float(prod)/norm_const for prod in unnormalized_prods]

            return alias_setup(normalized_prods)

    # Conduct preprocessing of transition probabilities when running Random walks
    def preprocess_transition_probs(self):
        graph = self.graph

        is_directed = self.is_directed

        alias_nodes = {}

        for node in graph.nodes():
            unnormalized_prods = [graph[node][nbr]['weight'] for nbr in sorted(graph.neighbors(node))]
            norm_const = sum(unnormalized_prods)
            normalized_prods = [float(prod)/norm_const for prod in unnormalized_prods]
            alias_nodes[node] = alias_setup(normalized_prods)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in graph.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in graph.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_edges = alias_edges 
        self.alias_nodes = alias_nodes

    # Repeat random walks for each nodes
    def simulate_walks(self, num_walks, walk_length):
        graph = self.graph
        walks = []
        nodes = list(graph.nodes())

        print('Walking...')
        for walk_iter in range(num_walks):
            print("Loop: ", str(walk_iter+1))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(length=walk_length, started_node=node))

        return walks

# Setup the urility lists to prepare for uniform sampling
def alias_setup(probs):
    length = len(probs)
    q = np.zeros(length)
    J = np.zeros(length, dtype=np.int)

    smaller = []
    larger = []

    for k, prod in enumerate(probs):
        q[k] = length * prod

        if q[k] < 1.0:
            smaller.append(k)
        else:
            larger.append(k)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

# Draw the sample
def alias_draw(J, q):
    length = len(J)

    k = int(np.floor(np.random.rand() * length))

    if np.random.rand() < q[k]:
        return k
    else:
        return J[k]
