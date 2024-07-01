#!/bin/bash

# sh sh/wdsinger_distmult.sh

data='WD-singer' 
score_func='distmult' 

CUDA_VISIBLE_DEVICES=0 python run.py -data $data -rel_reason -batch 256 -init_dim 150 -gcn_dim 150 -embed_dim 150 -gcn_layer 2 -gcn_drop 0. -score_func $score_func -chan_drop 0.1 -rel_mask 0. -rel_norm -hid_drop 0.2
