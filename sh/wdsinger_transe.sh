#!/bin/bash

# sh sh/wdsinger_transe.sh

data='WD-singer' 
score_func='transe' 

CUDA_VISIBLE_DEVICES=0 python run.py -data $data -rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.3 -score_func $score_func -chan_drop 0.3 -rel_mask 0.1 -rel_norm -hid_drop 0.1
