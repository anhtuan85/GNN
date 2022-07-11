import dgl
import numpy as np
import torch

class DeepWalk(torch.nn.Module):
    def __init__(self, G, window_size= 10, embedding_size = 2, walks_per_vertex = 20, walk_length = 6):
        self.G = G    #dgl Graph
        self.window_size = window_size    #window size
        self.embedding_size = embedding_size    # embedding size
        self.walks_per_vertex = walks_per_vertex    # walks per vertex
        self.walk_length = walk_length  # walk length

    def forward(self, G):
        '''Deep walk algorithm'''
        maximum_length = 2*self.window_size + 1
        for _ in range(self.walks_per_vertex):
            all_nodes =  G.nodes()    # Torch Tensor
            rand_idx = torch.randperm(all_nodes.shape[0])   # Random shuffle nodes
            all_nodes = all_nodes[rand_idx]
            for vertex_i in all_nodes:
                walk = self.randomWalk(vertex_i)

    def randomWalk(self, vertex_i):
        '''Random Walk frorm vertex_i'''
        idx =  0
        curr = vertex_i    #current node
        walk = -1 * torch.ones(self.walk_length)
        while idx < self.walk_length:
            walk[idx] = curr
            neighbors = self.G.out_edges(curr)[1]     #list all neighbors of current node
            
            if len(neighbors) == 0:    #for directed graph if we got leaf node
                return walk[:idx+1]

            curr = neighbors[torch.randperm(neighbors.shape[0])][0]
            idx += 1
        return walk
