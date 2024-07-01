#!/bin/bash

# sh sh/fb100_distmult.sh

data='FB15k-237' 
score_func='distmult' 

CUDA_VISIBLE_DEVICES=0 python run.py -data $data -rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0. -score_func $score_func -chamix_dim 200 -relmix_dim 200  -rel_norm -hid_drop 0.3 
