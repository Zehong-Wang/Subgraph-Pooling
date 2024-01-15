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



class DomainData(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        #self.root = root
        super(DomainData, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])