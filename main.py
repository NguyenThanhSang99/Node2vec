import numpy as np
import node2vec 
import networkx as nx 
from gensim.models import Word2Vec

def get_graph(path, directed):
    print("Getting graph...")
    graph = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
    for e in graph.edges():
        graph[e[0]][e[1]]['weight'] = 1

    if not directed:
        graph = graph.to_undirected()
    
    return graph

def run_embeddings_model(walks):
    walks = [str(walk) for walk in walks]
    model = Word2Vec(walks, vector_size=128, window=10, min_count=0, sg=1, workers=4, epochs=20)
    return model


def main():
    p = 1
    q = 1
    number_walks = 5
    walk_lengths = 8
    in_path = "data/data.edgelist"
    out_path = "data/data.embeddings.npy"
    directed = False

    networkx_graph = get_graph(in_path, directed)
    node2vec_graph = node2vec.Graph(networkx_graph, directed, p, q)
    node2vec_graph.preprocess_transition_probs()
    walks = node2vec_graph.simulate_walks(number_walks, walk_lengths)
    model = run_embeddings_model(walks)
    model.wv.save_word2vec_format(out_path)

if __name__ == "__main__":
	main()