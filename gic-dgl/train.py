"Implementation is based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi"

import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, Coauthor, AmazonCoBuy
from dgl.transform import add_self_loop, remove_self_loop

from gic import GIC, Classifier
import scipy.sparse as sp
from collections import Counter
import random
from sklearn.preprocessing import OneHotEncoder
from statistics import mean,stdev

from utils import get_train_val_test_split, sample_per_class, remove_underrepresented_classes, count_parameters, _sample_mask

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def main(args):
    
    torch.manual_seed(1234)
    
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        data = load_data(args)
        features = torch.FloatTensor(data.features)
        
        
        
        labels = torch.LongTensor(data.labels)
        in_feats = features.shape[1]
        g = data.graph
        if args.dataset == 'cora':
            g.remove_edges_from(nx.selfloop_edges(g))
            g.add_edges_from(zip(g.nodes(), g.nodes()))
        g = DGLGraph(g)
        attr_matrix  = data.features
        labels = data.labels
        
    else:
        if args.dataset == 'physics':
            data = Coauthor('physics')
        if args.dataset == 'cs':
            data = Coauthor('cs')
        if args.dataset == 'computers':
            data = AmazonCoBuy('computers')
        if args.dataset == 'photo':
            data = AmazonCoBuy('photo')
          
        g = data
        g = data[0]
        attr_matrix  = g.ndata['feat']
        labels = g.ndata['label']
        
        features = torch.FloatTensor(g.ndata['feat'])
        
    
    ### LCC of the graph
    n_components=1
    sparse_graph = g.adjacency_matrix_scipy(return_edge_ids=False)
    _, component_indices = sp.csgraph.connected_components(sparse_graph)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]


    adj_matrix = sparse_graph[nodes_to_keep][:, nodes_to_keep]
    num_nodes = len(nodes_to_keep)
    g = adj_matrix
    g = DGLGraph(g)
    g = remove_self_loop(g)
    g = add_self_loop(g)
    g = DGLGraph(g)
    
    
    g.ndata['feat'] = attr_matrix[nodes_to_keep]
    features = torch.FloatTensor(g.ndata['feat'].float())
    if args.dataset == 'cora' or args.dataset == 'pubmed':
        features = features / (features.norm(dim=1)+ 1e-8)[:, None]
    g.ndata['label'] = labels[nodes_to_keep]
    labels = torch.LongTensor(g.ndata['label'])
    
    

    in_feats = features.shape[1]
     
    unique_l = np.unique(labels, return_counts=False)
    n_classes = len(unique_l)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    
    
    print('Number of nodes',n_nodes,'Number of edges', n_edges)
    
    
    
    enc = OneHotEncoder()
    enc.fit(labels.reshape(-1,1))
    ylabels = enc.transform(labels.reshape(-1,1)).toarray()
    

    for beta in [args.beta]:
        for K in [args.num_clusters]:
            for alpha in [args.alpha]:
                accs = []
                t_st = time.time()
                
                sets = "imbalanced"
                
                for k in range(2):  #number of differnet trainings
                    #print(k)
                    
                    random_state = np.random.RandomState()
                    if sets=="imbalanced":
                        train_idx, val_idx, test_idx = get_train_val_test_split(random_state,
                                                 ylabels,
                                                 train_examples_per_class=None, val_examples_per_class=None,
                                                 test_examples_per_class=None,
                                                 train_size=20*n_classes, val_size=30*n_classes, test_size=None)
                    elif sets=="balanced":
                        train_idx, val_idx, test_idx = get_train_val_test_split(random_state,
                                                 ylabels,
                                                 train_examples_per_class=20, val_examples_per_class=30,
                                                 test_examples_per_class=None,
                                                 train_size=None, val_size=None, test_size=None)
                    else:
                        ("No such set configuration (imbalanced/balanced)")


                    n_nodes = len(nodes_to_keep)
                    train_mask = np.zeros(n_nodes)
                    train_mask[train_idx] = 1
                    val_mask = np.zeros(n_nodes)
                    val_mask[val_idx] = 1
                    test_mask = np.zeros(n_nodes)
                    test_mask[test_idx] = 1
                    train_mask = torch.BoolTensor(train_mask)
                    val_mask = torch.BoolTensor(val_mask)
                    test_mask = torch.BoolTensor(test_mask)
                    
                    """
                    Planetoid Split for CORA, CiteSeer, PubMed
                    train_mask = torch.BoolTensor(data.train_mask)
                    val_mask = torch.BoolTensor(data.val_mask)
                    test_mask = torch.BoolTensor(data.test_mask)
                    train_mask2 = torch.BoolTensor(data.train_mask)
                    val_mask2 = torch.BoolTensor(data.val_mask)
                    test_mask2 = torch.BoolTensor(data.test_mask)
                    """
                    
                    if args.gpu < 0:
                        cuda = False

                    else:
                        cuda = True
                        torch.cuda.set_device(args.gpu)
                        features = features.cuda()
                        labels = labels.cuda()
                        train_mask = train_mask.cuda()
                        val_mask = val_mask.cuda()
                        test_mask = test_mask.cuda()
                        
                    
                    gic = GIC(g,
                              in_feats,
                              args.n_hidden,
                              args.n_layers,
                              nn.PReLU(args.n_hidden),
                              args.dropout,
                              K,
                              beta, 
                              alpha
                              )

                    if cuda:
                        gic.cuda()

                    gic_optimizer = torch.optim.Adam(gic.parameters(),
                                                     lr=args.gic_lr,
                                                     weight_decay=args.weight_decay)

                    # train GIC
                    cnt_wait = 0
                    best = 1e9
                    best_t = 0
                    dur = []

                    

                    for epoch in range(args.n_gic_epochs):
                        gic.train()
                        if epoch >= 3:
                            t0 = time.time()

                        gic_optimizer.zero_grad()
                        loss = gic(features)
                        #print(loss)
                        loss.backward()
                        gic_optimizer.step()

                        if loss < best:
                            best = loss
                            best_t = epoch
                            cnt_wait = 0
                            torch.save(gic.state_dict(), 'best_gic.pkl')
                        else:
                            cnt_wait += 1

                        if cnt_wait == args.patience:
                            #print('Early stopping!')
                            break

                        if epoch >= 3:
                            dur.append(time.time() - t0)

                        #print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                              #"ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                            #n_edges / np.mean(dur) / 1000))
                    
                    # train classifier
                    #print('Loading {}th epoch'.format(best_t))
                    gic.load_state_dict(torch.load('best_gic.pkl'))
                    embeds = gic.encoder(features, corrupt=False)
                    embeds = embeds / (embeds+ 1e-8).norm(dim=1)[:, None]
                    embeds = embeds.detach()
                    
                    
                    
                    # create classifier model 
                    classifier = Classifier(args.n_hidden, n_classes)
                    if cuda:
                        classifier.cuda()

                    classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                                            lr=args.classifier_lr,
                                                            weight_decay=args.weight_decay)


                    dur = []
                    best_a = 0
                    cnt_wait = 0
                    for epoch in range(args.n_classifier_epochs):
                        classifier.train()
                        if epoch >= 3:
                            t0 = time.time()

                        classifier_optimizer.zero_grad()
                        preds = classifier(embeds)
                        loss = F.nll_loss(preds[train_mask], labels[train_mask])
                        loss.backward()
                        classifier_optimizer.step()

                        if epoch >= 3:
                            dur.append(time.time() - t0)

                        acc = evaluate(classifier, embeds, labels, val_mask) #+ evaluate(classifier, embeds, labels, train_mask)

                        if acc > best_a and epoch > 100:
                            best_a = acc
                            best_t = epoch

                            torch.save(classifier.state_dict(), 'best_class.pkl')

                        #print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                              #"ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                            #acc, n_edges / np.mean(dur) / 1000))

                    
                    acc = evaluate(classifier, embeds, labels, test_mask)
                    accs.append(acc)

                    

                    
                
                print('=================== ',' alpha', alpha, ' beta ', beta, 'K', K)
                print(args.dataset, ' Acc (mean)', mean(accs),' (std)',stdev(accs))
                print('=================== time', int((time.time() - t_st)/60))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIC')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--gic-lr", type=float, default=1e-3,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--n-gic-epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=50,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument('--b', dest='beta', type=int, default=100, help='')
    parser.add_argument('--c', dest='num_clusters', type=int, default=128, help='')
    parser.add_argument('--a', dest='alpha', type=float, default=0.5, help='')
    
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    #print(args)
    
    main(args)
