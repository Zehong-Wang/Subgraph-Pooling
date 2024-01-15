
import os.path as osp
from sklearn.decomposition import PCA
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



class BlogDomainData(InMemoryDataset):
    def __init__(self,root,name,use_pca=False,pca_dim=1000,transform=None,pre_transform=None,pre_filter=None):
        self.name=name
        self.use_pca = use_pca
        self.dim = pca_dim
        super(BlogDomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}.mat']
    
    @property
    def processed_file_names(self):
        if self.use_pca:
            return [f'data_pca_{self.dim}.pt']
        else:
            return ['data.pt']

    def feature_compression(self,features):
        """Preprcessing of features"""
        features = features.toarray()
        feat = sp.lil_matrix(PCA(n_components=self.dim, random_state=0).fit_transform(features))
        return feat.toarray()

    def process(self):
        net = sio.loadmat(osp.join(self.raw_dir, self.name+'.mat'))
        features, adj, labels = net['attrb'], net['network'], net['group']
        if not isinstance(features, sp.lil_matrix):
            features = sp.lil_matrix(features)

        if self.use_pca:
            features = self.feature_compression(features)
            features = torch.from_numpy(features).to(torch.float)
        else:
            features = features.todense().astype(float)
            features = torch.from_numpy(features).to(torch.float)
        if not isinstance(adj,sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        indices = np.vstack([adj.row, adj.col])
        edge_index = torch.tensor(indices,dtype=torch.long)

        y = torch.from_numpy(np.array(labels)).to(torch.long)
        y = torch.where(y >= 1)[1]
        data_list = []
        graph = Data(x=features, edge_index=edge_index,y=y)

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
