#!/bin/bash

# sh sh/nell23k_conve_restore.sh

data='NELL23K' 
score_func='conve' 

CUDA_VISIBLE_DEVICES=0 python restore.py -data $data -restore -name 'nell23k_conve' -rel_reason -pre_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.1 -score_func $score_func -chan_drop 0.1 -rel_mask 0.1 -rel_norm -hid_drop 0.3
