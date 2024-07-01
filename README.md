# HoGRN
Thank you for your interest in our work!
This is our Pytorch implementation for the paper:

>Weijian Chen, Yixn Cao, Fuli Feng, Xiangnan He, and Yongdong Zhang. [HoGRN: Explainable Sparse Knowledge Graph Completion via High-order Graph Reasoning Network](https://arxiv.org/abs/2207.07503). In IEEE Transactions on Knowledge and Data Engineering.

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@article{HoGRN,
  author       = {Weijian Chen and
                  Yixin Cao and
                  Fuli Feng and
                  Xiangnan He and
                  Yongdong Zhang},
  title        = {Explainable Sparse Knowledge Graph Completion via High-order Graph
                  Reasoning Network},
  journal      = {CoRR},
  volume       = {abs/2207.07503},
  year         = {2022},
  url          = {https://doi.org/10.48550/arXiv.2207.07503},
  doi          = {10.48550/ARXIV.2207.07503}
}
```
## Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
* pytorch == 1.6.0
* torch-sparse == 0.4.3
* torch-cluster == 1.4.5
* torch-scatter == 2.0.6
* numpy == 1.16.3
* ordered-set == 3.1

## Training and Evaluation
The main file is 'run.py', and description of commands has been clearly stated in this code (see the 'parser' function).
We provide a series of scripts in the "sh" folder to reproduce the results in our paper, which can be run with commands like ```sh sh/wdsinger_conve.sh```.


Running commands of HoGRN are as follows:
* NELL23K, HoGRN
```
CUDA_VISIBLE_DEVICES=0 python run.py -data 'WD-singer' -rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.3 -score_func 'conve' -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.1
```

Moreover, we provide relevant checkpoint files to quickly reproduce our experimental results through commands like ```sh sh/wdsinger_conve_restore.sh```. 
The full checkpoint files can be downloaded [here](https://drive.google.com/file/d/1Oo81Ge15FS2S5zHYtqb4jH4nn6HZegfA/view?usp=drive_link).

## Acknowledgment
Thanks to the following implementations:
* [CompGCN](https://github.com/malllabiisc/CompGCN)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
