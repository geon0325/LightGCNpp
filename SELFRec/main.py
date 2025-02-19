from SELFRec import SELFRec
from util.conf import ModelConf
import time
import argparse
import random
import torch
import numpy as np
import os

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp2018')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--model_name', type=str, default='SimGCL')
    parser.add_argument('--model_type', type=str, default='graph')
    parser.add_argument('--item_ranking', type=str, default='10,20,40')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.0001)
    
    # Common
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--lmbda2', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.2)
    
    # SGL
    parser.add_argument('--aug_type', type=int, default=1)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    
    # SimGCL
    parser.add_argument('--eps', type=float, default=0.1)
    
    # XSimGCL
    parser.add_argument('--l_star', type=int, default=1)
    
    # NCL
    parser.add_argument('--ssl_reg', type=float, default=1e-6)
    parser.add_argument('--proto_reg', type=float, default=1e-7)
    parser.add_argument('--hyper_layers', type=int, default=1)
    parser.add_argument('--num_clusters', type=int, default=2000)
    parser.add_argument('--ncl_alpha', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    
    parser.add_argument('--lmbda_ii', type=float, default=0.5)
    parser.add_argument('--lmbda_uu', type=float, default=0.5)
    parser.add_argument('--ego', type=int, default=1)
    parser.add_argument('--cos', type=int, default=1)
    parser.add_argument('--bpr', type=int, default=1)
    
    parser.add_argument('--lmbda_1', type=float, default=0.5)
    parser.add_argument('--lmbda_2', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--num_neg', type=int, default=1)
    
    parser.add_argument('--left_norm', type=float, default=0.5)
    parser.add_argument('--right_norm', type=float, default=0.5)

    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--neg_agg', type=str, default='max')
    
    # LightGCN-Sharp
    parser.add_argument('--init', type=int, default=0)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--lr_param', type=float, default=0.1)
    parser.add_argument('--ts_param', type=int, default=10)
    
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # GPU 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('embs', exist_ok=True)
    
    # Data path
    args.training_set = f'./dataset/{args.dataset}/train.txt'
    args.valid_set = f'./dataset/{args.dataset}/valid.txt'
    args.test_set = f'./dataset/{args.dataset}/test.txt'
    
    s = time.time()
    rec = SELFRec(args)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
