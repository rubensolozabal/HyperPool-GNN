# Description: Main script to launch experiments on Hyperbolic projection pooling for GNNs.
# Created by Ruben Solozabal on 01.11.2023.
# Last modified on 01.11.2023.
# Python version: 3.11

import argparse

import torch
import numpy as np
import torch_geometric.transforms as T

from utils import *
from datasets import get_planetoid_dataset
from train_eval import run
from models import get_model



def main():

    # Hyperparameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--logger', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--update_freq', type=int, default=50)
    parser.add_argument('--hyperparam', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0)     # Uniformity parameter
    parser.add_argument('--regularization', type=str, default=None)     # Regularization to apply
    parser.add_argument('--hyperbolicity', type=float, default=0)     # Regularization parameter
    parser.add_argument('--path_results', type=str, default="results/")
    args = parser.parse_args()

    # Set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Get dataset
    kwargs_dataset = {
        'model_name': args.model,
        'dataset': args.dataset, 
        'normalize_features': args.normalize_features, 
        'split': args.split,
    }
    dataset = get_planetoid_dataset(**kwargs_dataset)

    # Get model
    kwargs_model = {
        'model_name': args.model,
        'dataset': dataset, 
        'hidden_dim': args.hidden, 
        'dropout': args.dropout,
        'hyperbolicity': args.hyperbolicity,
    }
    model = get_model(**kwargs_model)

    # Launch runs
    kwargs_run = {
        'dataset': dataset, 
        'model': model,
        'args': args,
        'device': device,
    }

    if args.hyperparam == 'eps':
        for param in np.logspace(-3, 0, 10, endpoint=True):
            print(f"{args.hyperparam}: {param}")
            kwargs_run[args.hyperparam] = param
            results = run(**kwargs_run)
    elif args.hyperparam == 'update_freq':
        for param in [4, 8, 16, 32, 64, 128]:
            print(f"{args.hyperparam}: {param}")
            kwargs_run[args.hyperparam] = param
            results = run(**kwargs_run)
    else:
        results = run(**kwargs_run)
        pass

    # Save results
    import json
    with open(args.path_results + str(args.logger)+'_results.json', 'w') as outfile:
        json.dump(results, outfile)

if __name__ == '__main__':
    main()
