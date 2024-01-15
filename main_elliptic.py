import os
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
from utils import get_ood_dataset, seed_everything, combine_dicts, get_pooling_graph, get_scheduler, get_device
from eval import evaluate



def get_args():
    parser = argparse.ArgumentParser('Subgraph Pooling')

    # General Config
    parser.add_argument('--source_target', type=str, default='elliptic', help='Set the source and target datasets')
    parser.add_argument('--use_params', action='store_true', help='Whether to use the params')
    parser.add_argument('--param_path', type=str, default='params', help='The path of params')

    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--device', type=int, default=0)

    # Model Config
    parser.add_argument('--backbone', type=str, default='gat', choices=['gcn', 'gat', 'sage', 'sgc'])
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
    train_loader, valid_loader, test_loader = get_ood_dataset(dataset, domain)
    for i, data in enumerate(train_loader):
        train_loader[i].k_hop_edge_index, train_loader[i].k_hop_edge_attr = get_pooling_graph(data, params)
    for i, data in enumerate(valid_loader):
        valid_loader[i].k_hop_edge_index, valid_loader[i].k_hop_edge_attr = get_pooling_graph(data, params)
    for i, data in enumerate(test_loader):
        test_loader[i].k_hop_edge_index, test_loader[i].k_hop_edge_attr = get_pooling_graph(data, params)

    for i, _ in enumerate(train_loader):
        train_loader[i].to(device)
    for i, _ in enumerate(valid_loader):
        valid_loader[i].to(device)
    for i, _ in enumerate(test_loader):
        test_loader[i].to(device)

    return train_loader, valid_loader, test_loader


def train(encoder, loader, optimizer, scheduler=None):
    encoder.train()

    total_loss = 0
    for data in loader:
        z = encoder.encode(data.x, data.edge_index, data.edge_attr)
        z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
        pred = encoder.predict(z).log_softmax(dim=-1)

        pred = pred[data.y_mask]
        y = data.y[data.y_mask]
        y = F.one_hot(y, num_classes=data.num_classes).float()

        loss = F.cross_entropy(pred, y)
        total_loss += loss
    total_loss = total_loss / len(loader)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return total_loss.item()


def test(encoder, train_loader, valid_loader, test_loader, metric='acc'):
    encoder.eval()
    accs = []
    train_y, train_pred, valid_y, valid_pred = [], [], [], []
    for i, data in enumerate(train_loader):
        z = encoder.encode(data.x, data.edge_index, data.edge_attr)
        z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
        pred = encoder.predict(z)
        train_y.append(data.y[data.y_mask])
        train_pred.append(pred[data.y_mask])
    acc_train = evaluate(torch.cat(train_pred, dim=0), torch.cat(train_y, dim=0), metric) * 100

    for i, data in enumerate(valid_loader):
        z = encoder.encode(data.x, data.edge_index, data.edge_attr)
        z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
        pred = encoder.predict(z)
        valid_y.append(data.y[data.y_mask])
        valid_pred.append(pred[data.y_mask])
    acc_val = evaluate(torch.cat(valid_pred, dim=0), torch.cat(valid_y, dim=0), metric) * 100

    accs += [acc_train] + [acc_val]

    test_y, test_pred = [], []
    for i, data in enumerate(test_loader):
        z = encoder.encode(data.x, data.edge_index, data.edge_attr)
        z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
        pred = encoder.predict(z)
        test_y.append(data.y[data.y_mask])
        test_pred.append(pred[data.y_mask])
        if i % 4 == 0 or i == len(test_loader) - 1:
            acc_test = evaluate(torch.cat(test_pred, dim=0), torch.cat(test_y, dim=0), metric) * 100
            accs += [acc_test]
            test_y, test_pred = [], []

    return accs


def main(params):
    assert params['source_target'] in ['elliptic']

    params['source'] = params['target'] = 'elliptic'
    params['source_domain'] = params['target_domain'] = 0

    if params['use_params']:
        param_path = osp.join(params['param_path'], f"{params['source_target']}.yaml")
        with open(param_path, 'r') as f:
            default_params = yaml.safe_load(f)
        params.update(default_params[params['backbone']][params['sampling']])

        print('The updated params')
        print(params)
        print()

    device = get_device(params)

    if params['target'] in ['elliptic']:
        metric = 'f1'
    else:
        raise NotImplementedError('Metric not implemented')

    results = []

    for seed in params['seed']:
        seed_everything(seed)
    
        train_loader, valid_loader, test_loader = process_data(params['target'], params['target_domain'], device, params)

        pretrain_encoder = GNN_SP(
            input_dim=train_loader[0].x.shape[1],
            hidden_dim=params['hidden_dim'],
            output_dim=train_loader[0].num_classes,
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
            loss = train(pretrain_encoder, train_loader, pretrain_optimizer, scheduler=pretrain_scheduler)
            values = test(pretrain_encoder, train_loader, valid_loader, test_loader, metric)
            tmp_result = {
                'loss': loss, 'epoch': epoch,
                f'train_{metric}': values[0], f'val_{metric}': values[1],
                f'test_{metric}': values[2:], f'test_full_{metric}': np.mean(values[2:])
            }

            if values[1] >= best_result.get(f'val_{metric}', 0):
                best_result = tmp_result

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