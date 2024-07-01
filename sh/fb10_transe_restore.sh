#!/bin/bash

# sh sh/fb10_transe_restore.sh

data='FB15K-237-10' 
score_func='transe' 

CUDA_VISIBLE_DEVICES=0 python restore.py -data $data -restore -name 'fb10_transe' -rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func $score_func -chan_drop 0.1 -rel_norm -hid_drop 0.2 -sim_decay 1e-6 
