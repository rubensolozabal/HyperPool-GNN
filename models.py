import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCN2Conv


def get_model(model_name, dataset, hidden_dim, dropout):
    if model_name == 'GCN':
        model = GCN(dataset, hidden_dim, dropout)
    elif model_name == 'GCN2':
        model = GCN2(dataset, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.6)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    return model




class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, dropout):
        super(GCN, self).__init__()
        self.crd = CRD(dataset.num_features, hidden_dim, dropout)
        self.cls = CLS(hidden_dim, dataset.num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x
    
class GCN2(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)