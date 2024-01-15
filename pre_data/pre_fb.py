
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import Data, DataLoader
from torch_geometric import transforms as T

import scipy.sparse as sp
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io as sio
from sklearn.preprocessing import label_binarize
from ogb.nodeproppred import NodePropPredDataset

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url

from torch_geometric.datasets import MixHopSyntheticDataset
from torch_geometric.transforms import NormalizeFeatures

from os import path

import pickle as pkl

from torch_sparse import SparseTensor

import scipy.io
import scipy.sparse
import torch
import csv
import json
from os import path


def load_fb100(data_dir, filename):
    mat = scipy.io.loadmat(f'{data_dir}/{filename}.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_nc_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset
        Returns NCDataset
    """
    if dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Caltech36', 'Brown11', 'Yale4', 'Texas80',
                                'Bingham82', 'Duke14', 'Princeton12', 'WashU32', 'Brandeis99', 'Carnegie49'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(data_dir, sub_dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_fb100_dataset(data_dir, filename):
    feature_vals_all = np.empty((0, 6))
    for f in ['Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Caltech36', 'Brown11', 'Yale4', 'Texas80',
              'Bingham82', 'Duke14', 'Princeton12', 'WashU32', 'Brandeis99', 'Carnegie49']:
        A, metadata = load_fb100(data_dir, f)
        metadata = metadata.astype(np.int64)
        feature_vals = np.hstack(
            (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        feature_vals_all = np.vstack(
            (feature_vals_all, feature_vals)
        )

    A, metadata = load_fb100(data_dir, filename)
    # dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int64)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        # feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        feat_onehot = label_binarize(feat_col, classes=np.unique(feature_vals_all[:, col]))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    data = Data(edge_index=edge_index, x=node_feat, num_nodes=num_nodes)
    data.edge_attr = None
    data.num_classes = label.max().item() + 1
    data.y = torch.tensor(label)
    data.y = torch.where(data.y > 0, 1, 0)
    # dataset.graph = {'edge_index': edge_index,
    #                  'edge_attr': None,
    #                  'x': node_feat,
    #                  'num_nodes': num_nodes, 
    #                  'num_classes': label.max().item() + 1}
    # dataset.y = torch.tensor(label)
    # dataset.y = torch.where(dataset.y > 0, 1, 0)
    return data
