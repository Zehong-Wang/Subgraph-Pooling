import os.path as osp
import copy
import string 
import argparse

import yaml
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from model import GNN_SP
from utils import get_ood_dataset, seed_everything, combine_dicts, get_pooling_graph, target_sampling, source_sampling, DA_sampling, get_device, get_scheduler
from eval import evaluate


def get_args():
    parser = argparse.ArgumentParser('Subgraph Pooling')

    # General Config
    parser.add_argument('--source_target', type=str, default='facebook_1_2_3_10', help='Set the source and target datasets')
    parser.add_argument('--use_params', action='store_true', help='Whether to use the params')
    parser.add_argument('--param_path', type=str, default='params', help='The path of params')

    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--device', type=int, default=0)

    # Model Config
    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'sgc'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Sampling and Pooling Config
    parser.add_argument('--sampling', type=str, default='rw', choices=['k_hop', 'rw'], help='k_hop for SP, rw for SP++')
    parser.add_argument('--pooling', type=str, default='mean', choices=['gcn', 'mean', 'max', 'sum', 'attn'])
    parser.add_argument('--hops', type=int, default=2)
    # The following are only available for RW sampling
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--rw_mode', type=str, default='standard')
    parser.add_argument('--symm', action='store_true', default=True)
    parser.add_argument('--use_self_loop', action='store_true', default=True)

    # Training Config
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--use_scheduler', action='store_true')

    parser.add_argument('--pretrain_epochs', type=int, default=500, help='Epochs for pretraining')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Learning rate for pretraining')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0, help='Weight decay for pretraining')

    args = parser.parse_args()
    return vars(args)


def process_data(dataset, domain, device, params):
    psd_data = get_ood_dataset(dataset, domain)
    psd_data.k_hop_edge_index, psd_data.k_hop_edge_attr = get_pooling_graph(psd_data, params)
    return psd_data.to(device)


def train(encoder, data_list, optimizer, scheduler=None):
    encoder.train()

    total_loss = 0
    for data in data_list:
        z = encoder.encode(data.x, data.edge_index, data.edge_attr)
        z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
        pred = encoder.predict(z).log_softmax(dim=-1)

        y = F.one_hot(data.y, num_classes=data.num_classes).float()

        loss = F.cross_entropy(pred, y)
        
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return total_loss.item() / len(data_list)


def test(encoder, data, metric='acc'):
    encoder.eval()

    z = encoder.encode(data.x, data.edge_index, data.edge_attr)
    z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
    pred = encoder.predict(z)
    y = data.y if data.y.dim() == 1 else data.y.squeeze()

    value = evaluate(pred, y, metric) * 100

    return value


def main(params):
    source_target = params['source_target'].split('_')
    assert source_target[0] == 'facebook'
    params['source'] = params['target'] = source_target[0]
    params['source_domain'] = [eval(domain) for domain in source_target[1:-1]]
    params['target_domain'] = eval(source_target[-1])

    if params['use_params']:
        param_path = osp.join(params['param_path'], f"{params['source']}.yaml")
        with open(param_path, 'r') as f:
            default_params = yaml.safe_load(f)
        params.update(default_params[params['backbone']][params['sampling']])

        print('The updated params')
        print(params)
        print()

    device = get_device(params)
    metric = 'acc'

    results = []
    for seed in params['seed']:
        seed_everything(seed)

        src_data_list = [process_data(params['source'], source_domain, device, params) for source_domain in params['source_domain']]
        val_data_list = [process_data(params['source'], source_domain, device, params) for source_domain in [13, 14]]
        tgt_data = process_data(params['target'], params['target_domain'], device, params)

        pretrain_encoder = GNN_SP(
            input_dim=tgt_data.x.shape[1],
            hidden_dim=params['hidden_dim'],
            output_dim=tgt_data.y.max().item() + 1,
            activation=nn.PReLU,
            num_layers=params['num_layers'],
            backbone=params['backbone'],
            pooling=params['pooling'],
            dropout=params['dropout']
        ).to(device)

        pretrain_optimizer = AdamW(pretrain_encoder.parameters(), lr=params['pretrain_lr'], weight_decay=params['pretrain_weight_decay'])
        # pretrain_scheduler = get_scheduler(pretrain_optimizer, use_scheduler=params['use_scheduler'], epochs=params['pretrain_epochs'])
        pretrain_scheduler = None

        best_result = {}
        for epoch in range(1, params['pretrain_epochs'] + 1):
            loss = train(pretrain_encoder, src_data_list, pretrain_optimizer, scheduler=pretrain_scheduler)

            val_value = (test(pretrain_encoder, val_data_list[0], metric) + test(pretrain_encoder, val_data_list[1], metric)) / 2
            test_value = test(pretrain_encoder, tgt_data, metric)

            tmp_result = {
                'loss': loss, 'epoch': epoch, 
                f'val_{metric}': val_value, f'test_{metric}': test_value}
            
            if val_value >= best_result.get(f'val_{metric}', 0) or (val_value == best_result.get(f'val_{metric}', 0) and test_value > best_result.get(f'test_{metric}', 0)):
                best_result = tmp_result

            if epoch - best_result['epoch'] > params['early_stop']:
                break

            # print(f"DA Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_value:.4f}, Test: {test_value:.4f}")

        results.append(best_result)
        print(best_result)
        print()

    results = combine_dicts(results)
    print(results)
    print()




if __name__ == "__main__":
    params = get_args()
    print(params)
    print()
    
    main(params)