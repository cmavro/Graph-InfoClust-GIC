"Implementation based on https://github.com/PetarV-/DGI"
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import GIC, LogReg
from utils import process


from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import statistics 
import argparse

def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"
    
    score_matrix = np.dot(embeddings, embeddings.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    
    #print(preds_all, labels_all )
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--d', dest='dataset', type=str, default='cora',help='')
parser.add_argument('--b', dest='beta', type=int, default=100,help='')
parser.add_argument('--c', dest='num_clusters', type=float, default=128,help='')
parser.add_argument('--a', dest='alpha', type=float, default=0.5,help='')
parser.add_argument('--test_rate', dest='test_rate', type=float, default=0.1,help='')


args = parser.parse_args()
#print(args.accumulate(args.integers))

cuda0 = torch.cuda.is_available()#False

beta = args.beta
alpha = args.alpha
num_clusters = int(args.num_clusters)


dataset = args.dataset

# training params
batch_size = 1
nb_epochs = 2000
patience = 50
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 16
sparse = True
nonlinearity = 'prelu' # special name to separate parameters


         
torch.cuda.empty_cache()

roc0=[]
ap0=[]
roc1=[]
ap1=[]
roc100 = []
ap100 = []

for m in range(1):
    
    
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    
    adj_sparse = adj
    #print('Edges init',adj.getnnz())
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = process.mask_test_edges(adj, test_frac=args.test_rate, val_frac=0.05)
    adj = adj_train
    #print('Edges new',adj.getnnz())
    ylabels = labels
    
    features, _ = process.preprocess_features(features)


    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]


    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    #idx_test = torch.LongTensor(idx_test)

    
    if cuda0:
        #print('Using CUDA')
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()
        labels = labels.cuda()
        #idx_train = idx_train.cuda()
        #idx_val = idx_val.cuda()
        #idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    b_bce = nn.BCELoss()
    #xent = nn.CrossEntropyLoss()
    

    all_accs = []

    for beta in [args.beta]:
        print()
        for K in [int(args.num_clusters)]:
            #K = int(Kr * nb_nodes)
            for alpha in [args.alpha]:
                #print(m, alpha)

                model = GIC(nb_nodes,ft_size, hid_units, nonlinearity, num_clusters, beta)
                optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
                cnt_wait = 0
                best = 1e9
                best_t = 0
                val_best = 0
                
                if cuda0:
                #print('Using CUDA')
                    model.cuda()
                for epoch in range(nb_epochs):
                    model.train()
                    optimiser.zero_grad()

                    idx = np.random.permutation(nb_nodes)
                    shuf_fts = features[:, idx, :]

                    lbl_1 = torch.ones(batch_size, nb_nodes)
                    lbl_2 = torch.zeros(batch_size, nb_nodes)
                    lbl = torch.cat((lbl_1, lbl_2), 1)

                    

                    if cuda0:
                        shuf_fts = shuf_fts.cuda()
                        lbl = lbl.cuda()
                        

                    logits, logits2  = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None, beta) 


                    loss = alpha* b_xent(logits, lbl)  + (1-alpha)*b_xent(logits2, lbl) 

                    if loss < best:
                        best = loss
                        best_t = epoch
                        cnt_wait = 0
                        torch.save(model.state_dict(), dataset+'-link.pkl')
                        
                    else:
                        cnt_wait += 1

                    if cnt_wait == patience:
                        #print('Early stopping!')
                        break

                    
                        

                    loss.backward()
                    optimiser.step()

                    
                model.load_state_dict(torch.load(dataset+'-link.pkl'))

                embeds, _,_, S= model.embed(features, sp_adj if sparse else adj, sparse, None, beta)
                embs = embeds[0, :]
                embs = embs / embs.norm(dim=1)[:, None]
                
                sc_roc, sc_ap = get_roc_score(test_edges, test_edges_false, embs.cpu().detach().numpy(), adj_sparse)
                #print(beta, K, alpha, sc_roc, sc_ap,flush=True)
                print('Dataset',args.dataset)
                print('alpha, beta, K:',alpha,beta,K)
                print('AUC', sc_roc, 'AP', sc_ap)
