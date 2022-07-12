from importlib.metadata import requires
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
import math

'''
    Reference source: https://github.com/dsgiitr/graph_nets
'''

def func_n(w, j):
    li=[w]
    while(w!=1):
        w = w//2
        li.append(w)

    li.reverse()
    
    return li[j]

class SimpleSkipGram(torch.nn.Module):
    def __init__(self, num_nodes, embedding_size, window_size):
        super(SimpleSkipGram, self).__init__()
        self.window_size = window_size
        self.num_nodes = num_nodes

        self.phi = nn.Parameter(torch.rand((num_nodes, embedding_size), requires_grad = True))
        self.prob_tensor = nn.Parameter(torch.rand((2*num_nodes, embedding_size), requires_grad=True))

    def forward(self, node_j, node_k):
        one_hot_vec = torch.zeros(self.num_nodes)

        one_hot_vec[int(node_j.item())] = 1
        w = self.num_nodes + int(node_k.item())
        h = torch.matmul(one_hot_vec, self.phi)
        p = torch.tensor([1.0])

        length_path = int(math.log(w, 2)) +1   #Length of path from root to leaf in binary tree
        for i in range(1, length_path - 1):
            mult = -1
            if(func_n(w, i+1)==2*func_n(w, i)): # Left child
                mult = 1
    
            p = p*sigmoid(mult*torch.matmul(self.prob_tensor[func_n(w,i)], h))
        
        return p