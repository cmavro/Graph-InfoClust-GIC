"""
Graph InfoClust in DGL
Implementation is based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from gcn import GCN

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        #features @ self.weight @ summary.t()
        return torch.matmul(features, torch.matmul(self.weight, summary))
    
class DiscriminatorK(nn.Module):
    def __init__(self, n_hidden):
        super(DiscriminatorK, self).__init__()

    def forward(self, features, summary):
        
        n, h = features.size()
        
        ####features =  features / features.norm(dim=1)[:, None]
        #features = torch.sum(features*summary, dim=1)
        
        #features = features @ self.weight @ summary.t()
        return torch.bmm(features.view(n, 1, h), summary.view(n, h, 1)) #torch.sum(features*summary, dim=1) 


class GIC(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, K, beta, alpha):
        super(GIC, self).__init__()
        self.n_hidden = n_hidden
        self.g=g
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.discriminator2 = Discriminator(n_hidden)
        self.discriminatorK = DiscriminatorK(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()
        self.K = K
        self.beta = beta
        self.cluster = Clusterator(n_hidden,K)
        self.alpha = alpha
        

    def forward(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        graph_summary = torch.sigmoid(positive.mean(dim=0))
        
        mu, r = self.cluster(positive, self.beta)
        
        
        cluster_summary = torch.sigmoid(r @ mu)
        
        pos_graph = self.discriminator(positive, graph_summary)
        neg_graph = self.discriminator(negative, graph_summary)
        

        l1 = self.loss(pos_graph, torch.ones_like(pos_graph)) 
        l2 = self.loss(neg_graph, torch.zeros_like(neg_graph)) 

        l = self.alpha*(l1+l2)
        
        
        pos_cluster = self.discriminatorK(positive, cluster_summary)
        neg_cluster = self.discriminatorK(negative, cluster_summary)
        
        
        l += (1-self.alpha)*(self.loss(pos_cluster, torch.ones_like(pos_cluster)) + self.loss(neg_cluster, torch.zeros_like(neg_cluster))) 
                          
        
        return l

def cluster(data, k, temp, num_iter, init, cluster_temp):
    '''
    pytorch (differentiable) implementation of soft k-means clustering. 
    Modified from https://github.com/bwilder0/clusternet
    '''
    cuda0 = torch.cuda.is_available()#False
    
    
    
    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init
    n = data.shape[0]
    d = data.shape[1]

    data = data / (data.norm(dim=1) + 1e-8)[:, None]
    
    for t in range(num_iter):
        #get distances between all data points and cluster centers
        
        mu = mu / mu.norm(dim=1)[:, None]
        dist = torch.mm(data, mu.transpose(0,1))
        
        
        #cluster responsibilities via softmax
        r = F.softmax(cluster_temp*dist, dim=1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = r.t() @ data
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
        
    
    
    r = F.softmax(cluster_temp*dist, dim=1)
    
    
    return mu, r

class Clusterator(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    Modified from https://github.com/bwilder0/clusternet
    '''
    def __init__(self, nout, K):
        super(Clusterator, self).__init__()

        
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout
        
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, embeds, cluster_temp, num_iter=10):
        
        mu_init, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = torch.tensor(cluster_temp), init = self.init)
        #self.init = mu_init.clone().detach()
        mu, r = cluster(embeds, self.K, 1, 1, cluster_temp = torch.tensor(cluster_temp), init = mu_init.clone().detach())
        
        return mu, r
    
    

    
class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)