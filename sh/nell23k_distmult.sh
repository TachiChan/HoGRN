#!/bin/bash

# sh sh/nell23k_distmult.sh

data='NELL23K' 
score_func='distmult' 

CUDA_VISIBLE_DEVICES=0 python run.py -data $data -rel_reason -batch 256 -init_dim 150 -gcn_dim 150 -embed_dim 150 -gcn_layer 2 -gcn_drop 0.3 -score_func $score_func -chan_drop 0.2 -hid_drop 0.2
