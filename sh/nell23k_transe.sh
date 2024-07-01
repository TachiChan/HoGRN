#!/bin/bash

# sh sh/nell23k_transe.sh

data='NELL23K' 
score_func='transe' 

CUDA_VISIBLE_DEVICES=0 python run.py -data $data -rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.2 -score_func $score_func -chan_drop 0.1 -rel_norm -hid_drop 0.2 -sim_decay 1e-5
