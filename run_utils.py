
import random
import argparse  
import numpy as np 
import torch

from lora import run_lora

try:
    import wandb
except:
    print('wandb not available')
    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--r', default=2, type=int)
    parser.add_argument('--alpha', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--dropout_rate', default=0.25, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'])
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+',help='a list of strings', default=['q', 'k', 'v']) 
    
    parser.add_argument('--wandb', action='store_true', default=False)
    
    args = parser.parse_args()

    return args
    

        
