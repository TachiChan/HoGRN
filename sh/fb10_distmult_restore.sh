#!/bin/bash

# sh sh/fb10_distmult_restore.sh

data='FB15K-237-10' 
score_func='distmult' 

CUDA_VISIBLE_DEVICES=0 python restore.py -data $data -restore -name 'fb10_distmult' -rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.3 -score_func $score_func -hid_drop 0.1 -sim_decay 1e-5 
