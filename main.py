import os
import os.path as osp
import copy
import yaml
import random
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from model import GNN_SP
from utils import get_ood_dataset, seed_everything, combine_dicts, get_pooling_graph, target_sampling, source_sampling, DA_sampling, sampling, CMD, random_string, flip_edges, get_device, get_scheduler
from eval import evaluate



def get_args():
    parser = argparse.ArgumentParser('Subgraph Pooling')

    # General Config
    parser.add_argument('--source_target', type=str, default='acm_dblp', help='Set the source and target datasets')
    parser.add_argument('--use_params', action='store_true', help='Whether to use the params')
    parser.add_argument('--param_path', type=str, default='params', help='The path of params')

    parser.add_argument('--freeze', action='store_true', help='Transfer Setting 1: Do not fine-tune the model. (Domain Adaptation)')
    parser.add_argument('--ft_last_layer', action='store_true', help='Transfer Setting 2: Fine-tune the last layer')
    parser.add_argument('--ft_whole_model', action='store_true', help='Transfer Setting 3: Fine-tune the whole model')

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
    parser.add_argument('--src_train_ratio', type=float, default=0.6, help='Ratio of training nodes in source dataset')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='Ratio of training nodes in target dataset')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--use_scheduler', action='store_true')

    parser.add_argument('--pretrain_epochs', type=int, default=500, help='Epochs for pretraining')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Learning rate for pretraining')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0, help='Weight decay for pretraining')
    parser.add_argument('--target_epochs', type=int, default=3000, help='Epochs for fine-tuning')
    parser.add_argument('--target_lr', type=float, default=1e-3, help='Learning rate for fine-tuning')
    parser.add_argument('--target_weight_decay', type=float, default=1e-5, help='Weight decay for fine-tuning')

    # Others
    parser.add_argument('--target_feature_noise', type=float, default=0.0)
    parser.add_argument('--target_edge_noise', type=float, default=0.0)
    parser.add_argument('--eval_disc', action='store_true', help='Whether to evaluate the discrepancy between source and target')
    parser.add_argument('--disc_verbose', type=int, default=1)

    args = parser.parse_args()
    return vars(args)



def process_data(dataset, domain, device, params):
    psd_data = get_ood_dataset(dataset, domain)
    psd_data.k_hop_edge_index, psd_data.k_hop_edge_attr = get_pooling_graph(psd_data, params)
    return psd_data.to(device)


def get_source_split(src_data, source, source_domain, src_train_ratio=0.6):
    if source in ['acm', 'dblp', 'de', 'en', 'es', 'fr', 'pt', 'ru', 'facebook', 'usa', 'brazil', 'europe', 'blog1', 'blog2'] or (source == 'arxiv' and source_domain != 0):
        return source_sampling(src_data, train_ratio=src_train_ratio)
    else:
        return {
            'train': src_data.src_train_mask,
            'valid': src_data.src_valid_mask,
            'test': src_data.src_test_mask
        }



def discrepancy(encoder, src_data, tgt_data, cmd, n_moments=3): 
    z1 = encoder.encode(src_data.x, src_data.edge_index, src_data.edge_attr)
    z1 = encoder.pooling(z1, src_data.k_hop_edge_index, src_data.get('k_hop_edge_attr', None))
    z2 = encoder.encode(tgt_data.x, tgt_data.edge_index, tgt_data.edge_attr)
    z2 = encoder.pooling(z2, tgt_data.k_hop_edge_index, tgt_data.get('k_hop_edge_attr', None))
    return cmd.mmatch(z1, z2, n_moments=n_moments)


def train(encoder, data, optimizer, split, scheduler=None):
    assert split is not None

    encoder.train()

    z = encoder.encode(data.x, data.edge_index, data.edge_attr)
    z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))

    pred = encoder.predict(z).log_softmax(dim=-1)

    y = F.one_hot(data.y, num_classes=data.num_classes).float()

    loss = F.cross_entropy(pred[split['train']], y[split['train']])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return loss.item()


def test(encoder, data, split, metric='acc'):
    encoder.eval()

    z = encoder.encode(data.x, data.edge_index, data.edge_attr)
    z = encoder.pooling(z, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
    pred = encoder.predict(z)
    y = data.y if data.y.dim() == 1 else data.y.squeeze()

    if split.get('train') is not None:
        train_value = evaluate(pred[split['train']], y[split['train']], metric) * 100
    else:
        train_value = 0
    if split.get('valid') is not None:
        val_value = evaluate(pred[split['valid']], y[split['valid']], metric) * 100
    else:
        val_value = 0
    if split.get('test') is not None:
        test_value = evaluate(pred[split['test']], y[split['test']], metric) * 100
    else:
        test_value = 0

    return train_value, val_value, test_value


def main(params):
    source_target = params['source_target'].split('_')

    if len(source_target) == 2:
        # If source and target domains are not available, set them to 0
        params['source'], params['source_domain'], params['target'], params['target_domain'] = source_target[0], 0, source_target[1], 0
    elif len(source_target) == 3:
        params['source'], params['source_domain'], params['target'], params['target_domain'] = source_target[0], eval(source_target[2]), source_target[1], eval(source_target[2])
    elif len(source_target) == 4:
        params['source'], params['source_domain'], params['target'], params['target_domain'] = source_target[0], eval(source_target[1]), source_target[2], eval(source_target[3])
    elif len(source_target) >= 5:
        assert source_target[0] == 'facebook'
        params['source'] = params['target'] = source_target[0]
        params['source_domain'] = [eval(domain) for domain in source_target[1:-1]]
        params['target_domain'] = eval(source_target[-1])
    else:
        raise NotImplementedError('Source and target not implemented')

    if params['use_params']:
        param_path = osp.join(params['param_path'], f"{params['source_target']}.yaml")
        with open(param_path, 'r') as f:
            default_params = yaml.safe_load(f)
        params.update(default_params[params['backbone']][params['sampling']])

        print('The updated params')
        print(params)
        print()

    device = get_device(params)

    if params['target'] in ['de', 'en', 'es', 'fr', 'pt', 'ru']:
        metric = 'auc'
    elif params['target'] in ['elliptic']:
        metric = 'f1'
    else:
        metric = 'acc'

    results = []
    freeze_loss_list = []
    freeze_test_acc_list = []
    ft_last_layer_test_acc_list = []
    
    if params['eval_disc']:
        disc_list = []
        cmd = CMD()

    for seed in params['seed']:
        seed_everything(seed)

        if params['source'] != params['target']:
            # ACM -> DBLP, DBLP -> ACM
            src_data = process_data(params['source'], params['source_domain'], device, params)
            tgt_data = process_data(params['target'], params['target_domain'], device, params)
        else:
            if params['source'] == 'arxiv' and params['source_domain'] != 0:
                # Arxiv-time
                src_data = process_data(params['source'], params['source_domain'], device, params)
                tgt_data = process_data(params['target'], params['target_domain'], device, params)
            else:
                # Cora-word, Cora-degree, Arxiv-degree
                data = process_data(params['source'], params['source_domain'], device, params)
                src_data = tgt_data = data

        if params['target_feature_noise'] != 0:
            tgt_data.x = (1 - params['target_feature_noise']) * tgt_data.x + params['target_feature_noise'] * torch.randn_like(tgt_data.x)
            print("Add Gaussian noise on nodes with level {} on target!".format(params['target_feature_noise']))

        if params['target_edge_noise'] != 0:
            tgt_data = flip_edges(tgt_data, p=params['target_edge_noise'])
            print('Randomly flip {} edges on target!'.format(params['target_edge_noise']))


        if params['source'] in ['usa', 'brazil', 'europe']:
            src_split = {'train': src_data.tgt_mask}
        else:
            src_split = get_source_split(src_data, params['source'], params['source_domain'], src_train_ratio=params['src_train_ratio'])

        tgt_split = target_sampling(tgt_data, train_ratio=params['train_ratio'])
        da_split = DA_sampling(tgt_data)

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
        pretrain_scheduler = None

        best_ft_whole_model_result = {}; best_ft_last_layer_result = {}; best_freeze_result = {}

        if params['eval_disc']:
            tmp_disc_list = []

        freeze_losses = []
        freeze_test_accs = []
        ft_last_layer_test_accs = []


        # Pretrain the model based on source dataset with source split
        for epoch in range(1, params['pretrain_epochs'] + 1):
            loss = train(pretrain_encoder, src_data, pretrain_optimizer, split=src_split, scheduler=pretrain_scheduler)

            # Transfer Setting 1: Do not fine-tune the model.
            if params['freeze']:
                values = test(pretrain_encoder, tgt_data, da_split, metric)
                # da_loss is the loss on source dataset
                tmp_freeze_result = {'freeze_loss': loss, f'freeze_val_{metric}': values[1], f'freeze_test_{metric}': values[2], 'freeze_epoch': epoch}
                freeze_losses.append(loss)
                freeze_test_accs.append(values[2])

                if values[1] >= best_freeze_result.get(f'freeze_val_{metric}', 0):
                    best_freeze_result = tmp_freeze_result
            
            if params['eval_disc'] and epoch % params['disc_verbose'] == 0:
                disc = discrepancy(pretrain_encoder, src_data, tgt_data, cmd).item()
                best_freeze_result['disc'] = disc
                tmp_disc_list.append(disc)

        freeze_loss_list.append(freeze_losses)
        freeze_test_acc_list.append(freeze_test_accs)

        if params['eval_disc']:
            disc_list.append(tmp_disc_list)


        # Transfer Setting 2: Fine-tune the last layer
        if params['ft_last_layer']:
            encoder = copy.deepcopy(pretrain_encoder)
            encoder.freeze_params()
            encoder.unfreeze_pred_head()
            if encoder.fine_tune_aggr:
                encoder.unfreeze_pred_head()

            target_optimizer = AdamW(encoder.parameters(), lr=params['target_lr'], weight_decay=params['target_weight_decay'])
            target_scheduler = get_scheduler(target_optimizer, use_scheduler=params['use_scheduler'], epochs=params['target_epochs'])

            for epoch in range(1, params['target_epochs'] + 1):
                loss = train(encoder, tgt_data, target_optimizer, split=tgt_split, scheduler=target_scheduler)

                if epoch % params['verbose'] == 0:
                    values = test(encoder, tgt_data, tgt_split, metric)

                    tmp_result = {'ft_last_layer_loss': loss, f'ft_last_layer_train_{metric}': values[0],
                                  f'ft_last_layer_val_{metric}': values[1], f'ft_last_layer_test_{metric}': values[2],
                                  'ft_last_layer_epoch': epoch}
                    ft_last_layer_test_accs.append(values[2])

                    if values[1] >= best_ft_last_layer_result.get(f'ft_last_layer_val_{metric}', 0):
                        best_ft_last_layer_result = tmp_result

                    if epoch - best_ft_last_layer_result['ft_last_layer_epoch'] > params['early_stop']:
                        break
            ft_last_layer_test_acc_list.append(ft_last_layer_test_accs)

        
        # Transfer Setting 3: Fine-tune the whole model
        if params['ft_whole_model']:
            encoder = copy.deepcopy(pretrain_encoder)

            target_optimizer = AdamW(encoder.parameters(), lr=params['target_lr'], weight_decay=params['target_weight_decay'])
            target_scheduler = get_scheduler(target_optimizer, use_scheduler=params['use_scheduler'], epochs=params['target_epochs'])

            for epoch in range(1, params['target_epochs'] + 1):
                loss = train(encoder, tgt_data, target_optimizer, split=tgt_split, scheduler=target_scheduler)

                if epoch % params['verbose'] == 0:
                    values = test(encoder, tgt_data, tgt_split, metric)

                    tmp_result = {'ft_whole_model_loss': loss, f'ft_whole_model_train_{metric}': values[0], f'ft_whole_model_val_{metric}': values[1],
                                  f'ft_whole_model_test_{metric}': values[2], 'ft_whole_model_epoch': epoch}

                    if values[1] >= best_ft_whole_model_result.get(f'ft_whole_model_val_{metric}', 0):
                        best_ft_whole_model_result = tmp_result

                    if epoch - best_ft_whole_model_result['ft_whole_model_epoch'] > params['early_stop']:
                        break

        best_result = {**best_ft_whole_model_result, **best_ft_last_layer_result, **best_freeze_result}
        print(best_result)
        print()

        results.append(best_result)

        # Clear GPU cache and memory cache
        del pretrain_encoder, src_data, tgt_data
        if params['ft_last_layer'] or params['ft_whole_model']:
            del encoder
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    results = combine_dicts(results)
    print(results)
    print()

    if params['eval_disc']:
        results['disc'] = disc_list
        results['freeze_loss'] = freeze_loss_list
        results['freeze_test_acc'] = freeze_test_acc_list
        results['ft_last_layer_test_acc'] = ft_last_layer_test_acc_list


if __name__ == "__main__":
    params = get_args()
    print(params)
    print()

    main(params)