import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        # Set dropout rate - this means that at each training iteration,
        # each node is exposed to a stochastically sampled neighborhood)
        self.dropout = dropout

        # Define a list of attention heads
        self.attentions = [GraphAttentionLayer(in_features=nfeat,
                                               out_features=nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=True) for _ in range(nheads)]

        # Add the attention heads as nn.Module attribute for GAT class
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)

        self.out_att = GraphAttentionLayer(in_features=nhid * nheads,
                                           out_features=nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        # Dropout rate of self.dropout if nodel in training mode
        x = F.dropout(x, self.dropout, training=self.training)
        # Get all attention heads output representation and concatenate to one long vector
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # Apply another dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply elu activation function
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

