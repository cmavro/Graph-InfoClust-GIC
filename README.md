# Graph-InfoClust-GIC [PAKDD 2021]
[PAKDD'21 version](https://www.springerprofessional.de/en/graph-infoclust-maximizing-coarse-grain-mutual-information-in-gr/19147684) Graph InfoClust: Maximizing Coarse-Grain Mutual Information in Graphs

[Preprint version](https://arxiv.org/abs/2009.06946) Graph InfoClust: Leveraging cluster-level node information for unsupervised graph representation learning


An unsupervised node representation learning method (to appear in PAKDD 2021).

# Overview
![](/images/GIC_overview.png?raw=true "")

GIC’s framework. (a) A fake input is created based on the real one. (b) Embeddings are computed for both inputs with a GNN-encoder. (c) The graph and cluster summaries are computed. (d) The goal is to discriminate between real and fake samples based on the computed summaries.

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
@misc{mavromatis2020graph,
    title={Graph InfoClust: Leveraging cluster-level node information for unsupervised graph representation learning},
    author={Costas Mavromatis and George Karypis},
    year={2020},
    eprint={2009.06946},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
or
```
@inproceedings{Mavromatis2021GraphIM,
  title={Graph InfoClust: Maximizing Coarse-Grain Mutual Information in Graphs},
  author={Costas Mavromatis and G. Karypis},
  booktitle={PAKDD},
  year={2021}
}
```
