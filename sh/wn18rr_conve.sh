#!/bin/bash

# sh sh/wn18rr_conve.sh

data='WN18RR' 
score_func='conve' 

CUDA_VISIBLE_DEVICES=0 python run.py -data $data -rel_reason -reason_type 'mixdrop2' -bias -batch 256 -init_dim 200 -gcn_dim 200 -embed_dim 200 -gcn_layer 1 -gcn_drop 0. -score_func $score_func -chamix_dim 300 -relmix_dim 300 -rel_norm -hid_drop 0.3 -hid_drop2 0.5 -feat_drop 0.1 -k_w 10 -k_h 20 -num_filt 250 -ker_sz 7
