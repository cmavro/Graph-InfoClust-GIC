# Graph-InfoClust-GIC
[Graph InfoClust](https://arxiv.org/abs/2009.06946): Leveraging cluster-level node information for unsupervised graph representation learning
https://arxiv.org/abs/2009.06946

An unsupervised node representation learning method (under review).

# Overview
![](/images/GIC_overview.png?raw=true "")

GIC’s framework. (a) A fake input is created based on the real one. (b) Embeddings are computed for bothinputs with a GNN-encoder. (c) The graph and cluster summaries are computed. (d) The goal is to discriminate betweenreal and fake samples based on the computed summaries.

## gic-dgl
GIC (node classification task) implemented in [Deep Graph Library](https://github.com/dmlc/dgl) (DGL) , which should be installed.
```
python train.py --dataset=[DATASET]
```

## GIC
GIC (link prediction, clustering, and visualization tasks) based on Deep Graph Infomax (DGI) original implementation.
```
python execute_link.py
```

# Cite
```
@article{mavromatis2020graph,
  title={Graph InfoClust: Leveraging cluster-level node information for unsupervised graph representation learning},
  author={Mavromatis, Costas and Karypis, George},
  journal={arXiv preprint arXiv:2009.06946},
  year={2020}
}
```
