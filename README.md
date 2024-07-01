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
The primary script is 'run.py', which includes a clear documentation of the commands within its 'parser' function.
For the reproduction of the results presented in our paper, we have provided a set of scripts in the 'sh' directory. These can be executed with commands such as ```sh sh/wdsinger_conve.sh```. 
Additionally, to facilitate the swift replication of our experimental outcomes, we have made available pertinent checkpoint files, which can be utilized via commands like ```sh sh/wdsinger_conve_restore.sh```.
The full checkpoint files can be downloaded [here](https://drive.google.com/file/d/1Oo81Ge15FS2S5zHYtqb4jH4nn6HZegfA/view?usp=drive_link).
 
## Acknowledgment
Thanks to the following implementations:
* [CompGCN](https://github.com/malllabiisc/CompGCN)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
