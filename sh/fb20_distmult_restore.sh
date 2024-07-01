#!/bin/bash

# sh sh/fb20_distmult_restore.sh

data='FB15K-237-20' 
score_func='distmult' 

CUDA_VISIBLE_DEVICES=0 python restore.py -data $data -restore -name 'fb20_distmult' -rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.2 -score_func $score_func -chamix_dim 400 -relmix_dim 400  -rel_norm -hid_drop 0.3
