import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import MLP, MessagePassing, SAGEConv, GATConv, GCNConv, GINConv, SGConv, APPNP, global_add_pool, global_mean_pool, global_max_pool
from aggr import Aggregator, GCNAggr


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation, num_layers, backbone='sage',
                 normalize='none', dropout=0.0):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = normalize

        self.activation = activation()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        dims = [input_dim] + [hidden_dim] * num_layers

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            if backbone == 'sage':
                self.layers.append(SAGEConv(in_dim, out_dim, aggr='mean', normalize=True, root_weight=True))
            elif backbone == 'gat':
                self.layers.append(GATConv(in_dim, out_dim // 4, heads=4))
            elif backbone == 'gcn':
                self.layers.append(GCNConv(in_dim, out_dim, ))
            elif backbone == 'gin':
                self.layers.append(GINConv(nn.Linear(in_dim, out_dim)))
            self.norms.append(nn.BatchNorm1d(out_dim))

        self.lin = nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return self.predict(z)

    def encode(self, x, edge_index, edge_attr=None):
        z = x

        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_attr)
            # z = self.norms[i](z)
            z = self.activation(z)
            z = self.dropout(z)
        return z

    def predict(self, x):
        return self.lin(x)

    def reset_lin(self, num_classes):
        device = next(self.lin.parameters()).device

        self.lin = nn.Linear(self.hidden_dim, num_classes).to(device)
        self.lin.reset_parameters()

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_params(self):
        for param in self.parameters():
            param.requires_grad = True


class GNN_SP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation, num_layers, backbone='sage',
                 pooling='mean', normalize='none', dropout=0.0):
        super(GNN_SP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = normalize
        self.pooling_method = pooling
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        dims = [input_dim] + [hidden_dim] * num_layers
        self._init_layers(backbone, dims)
        self._init_pooling(pooling)
        self._init_lin()
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def _init_layers(self, backbone, dims):
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            if backbone == 'sage':
                self.layers.append(SAGEConv(in_dim, out_dim, aggr='mean', normalize=True, root_weight=True))
            elif backbone == 'gat':
                self.layers.append(GATConv(in_dim, out_dim // 4, heads=4))
            elif backbone == 'gcn':
                self.layers.append(GCNConv(in_dim, out_dim, ))
            elif backbone == 'sgc':
                self.layers.append(SGConv(self.input_dim, self.hidden_dim, K=self.num_layers))
                break
            elif backbone == 'appnp':
                self.layers.append(APPNP(K=self.num_layers, alpha=0.1))
                break
            elif backbone == 'mlp': 
                self.layers.append(nn.Linear(in_dim, out_dim))
            else:
                raise NotImplementedError('Backbone not implemented')

    def _init_pooling(self, pooling):
        if pooling in ['gcn']:
            self.aggr = GCNAggr(improved=False, cached=False)
            self.fine_tune_aggr = False
        elif pooling in ['sum', 'mean', 'max']:
            self.aggr = Aggregator(pooling)
            self.fine_tune_aggr = False
        elif pooling in ['attn']:
            self.aggr = GATConv(self.hidden_dim, self.hidden_dim // 1, heads=1)
            self.fine_tune_aggr = True
            if self.backbone == 'appnp':
                self.aggr = GATConv(self.input_dim, self.input_dim // 1, heads=1)
                self.fine_tune_aggr = True
        elif pooling in ['linear']:
            self.aggr = SAGEConv(self.hidden_dim, self.hidden_dim, aggr='mean', normalize=True, root_weight=True)
            self.fine_tune_aggr = True
        else:
            raise NotImplementedError('Pooling not implemented')

    def _init_lin(self):
        if self.backbone in ['appnp']:
            self.lin = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.lin = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x, edge_index, k_hop_edge_index, edge_attr=None, k_hop_edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        z = self.pooling(z, k_hop_edge_index, k_hop_edge_attr)
        return self.predict(z)

    def encode(self, x, edge_index, edge_attr=None):
        z = x

        for i, conv in enumerate(self.layers):
            if self.backbone == 'mlp':
                z = conv(z)
            else:
                z = conv(z, edge_index, edge_attr)
            # z = self.norms[i](z)
            z = self.activation(z)
            z = self.dropout(z)
        return z
    
    def encode_layer_by_layer(self, x, edge_index, edge_attr=None, layer=0):
        z = x

        for i, conv in enumerate(self.layers):
            if i == layer:
                if self.backbone == 'mlp':
                    z = conv(z)
                else:
                    z = conv(z, edge_index, edge_attr)
                # z = self.norms[i](z)
                z = self.activation(z)
                z = self.dropout(z)
        return z

    def pooling(self, x, edge_index, edge_attr=None):
        return self.aggr(x, edge_index, edge_attr)

    def predict(self, x):
        return self.lin(x)

    def reset_lin(self, num_classes):
        device = next(self.lin.parameters()).device

        self.lin = nn.Linear(self.hidden_dim, num_classes).to(device)
        self.lin.reset_parameters()

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_pred_head(self):
        for param in self.lin.parameters():
            param.requires_grad = True

    def unfreeze_pooling(self):
        for param in self.aggr.parameters():
            param.requires_grad = True