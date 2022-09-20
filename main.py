import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def random_walk(node_matrix, node, num_walk, p, q):
    previous_node = np.random.choice(pos_list(node_matrix, node)) 
    walk_list = [node]

    for i in range(num_walk):
        next_node = next_step(node_matrix, node, previous_node, p, q)
        walk_list.append(next_node)
        node = next_node
        previous_node = node
    return walk_list

def next_step(node_matrix, node, previous_node, p, q):
    positive = pos_list(node_matrix, node)
    li = np.array([])
    for pos in positive:
        if pos == previous_node:
            li = np.append(li, 1 / p)
        elif pos in pos_list(node_matrix, node):
            li = np.append(li, 1)
        else:
            li = np.append(li, 1 / q)
    probability = li / li.sum()
    return np.random.choice(positive, 1, p=probability)[0]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def pos_list(node_matrix, node):
    return np.nonzero(node_matrix[node])[1]

def neg_list(node_matrix, node):
    return np.where(node_matrix[node]==0)[1]

def node2vec(node_matrix, dimentional, num_epoch, length, learning_rate, k, p, q, num_neg):
    embed = np.random.random((node_matrix.shape[0], dimentional))
    for epoch in range(num_epoch):
        print("Epoch {}".format(epoch + 1))
        for v in np.arange(node_matrix.shape[0]):
            walk = random_walk(node_matrix, v, length - 1, p, q) 
            for idx in range(length - k):
                not_neg_list = np.append(walk[max(0, idx - k):idx + k], pos_list(node_matrix, walk[idx]))
                neg_list = list(set(np.arange(node_matrix.shape[0])) - set(not_neg_list))
                random_neg = np.random.choice(neg_list, num_neg, replace=False)
                for pos in range(idx + 1, idx + k + 1):
                    if walk[idx] != walk[pos]:
                        pos_embed = embed[walk[pos]]
                        embed[walk[idx]] -= learning_rate * (sigmoid(np.dot(embed[walk[idx]], pos_embed)) - 1) * pos_embed

                for neg in random_neg:
                    neg_embed = embed[neg]
                    embed[walk[idx]] -= learning_rate * (sigmoid(np.dot(embed[walk[idx]], neg_embed))) * neg_embed
    return embed

def main():
    graph = nx.read_gml('data/data.gml')
    node_matrix = nx.to_numpy_matrix(graph , nodelist=graph.nodes())
    print(node_matrix)

    embedding = node2vec(node_matrix=node_matrix, dimentional=2,num_epoch=50,length=3,learning_rate=0.01,
                 k=2,p=2,q=2,num_neg=5)
    print(embedding)
    plt.plot(embedding, 'o')
    plt.show()

if __name__ == '__main__':
    main()