#!/bin/bash

# sh sh/fb50_distmult_restore.sh

data='FB15K-237-50' 
score_func='distmult' 

CUDA_VISIBLE_DEVICES=0 python restore.py -data $data -restore -name 'fb50_distmult' -rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0. -score_func $score_func -chamix_dim 200 -relmix_dim 200  -rel_norm -hid_drop 0.3 
