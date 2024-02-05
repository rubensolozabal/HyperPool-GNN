import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SplineConv, SAGEConv, GATv2Conv, GINEConv, HGTConv, TransformerConv
from torch_geometric.nn import GCN2Conv
import manifolds
from utils.math_utils import temperature_scaled_logminexp as temperature_scaled_logminexp
from torch_geometric.nn.models import MLP, GIN


def get_model(model_name, dataset, hidden_dim, dropout, hyperbolicity=0):
    if model_name == 'GCN':
        # model = wrapperGCN(dataset, hidden_channels=64, num_layers=6, dropout=0.6)
        # model = GCN(in_channels=dataset.num_features, hidden_channels=hidden_dim, out_channels=dataset.num_classes, num_layers=2, dropout=dropout)
        # model = GCN(dataset, hidden_channels=64, num_layers=2, dropout=0.5)
        # model = GCN_SSL(dataset, hidden_dim, dropout)
        model = GCN(dataset, hidden_dim, num_layers=2, dropout=0.5)
    elif model_name == 'GAT':
        model = GAT_v2(dataset, hidden_channels=16, num_layers=2, heads=8, dropout=0.6)
    # elif model_name == 'GIN':
    #     model = GIN_v2(dataset, hidden_channels=64, num_layers=2, dropout=0.6)
    elif model_name == 'Spline':
        model = Spline(dataset, hidden_dim, num_layers=2, dropout=0.6)
    elif model_name == 'SAGE':
        model = SAGE(dataset, hidden_dim, num_layers=2, dropout=0.6)
    elif model_name == 'Transformer':
        model = Transformer(dataset, hidden_channels=16, heads=2, num_layers=2, dropout=0.6)
    #################################
    elif model_name == 'GCN2':
        model = GCN2(dataset, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.6)
    elif model_name == 'GCN2_BP':
        model = GCN2_BP(dataset, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.6)
    elif model_name == 'GCN2_HBP':
        model = GCN2_HBP(dataset, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.6, curvature=hyperbolicity)
    elif model_name == 'GCN2_HBPU':
        model = GCN2_HBPU(dataset, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.6, curvature=hyperbolicity)
    elif model_name == 'GCN2_HLBP':
        model = GCN2_HLBP(dataset, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.6, curvature=hyperbolicity)
    #################################
    elif model_name == 'GCN_HBP':
        model = GCN_HBP(dataset, hidden_dim, num_layers=2, curvature=hyperbolicity, dropout=0.6)
    elif model_name == 'SAGE_HBP':
        model = SAGE_HBP(dataset, hidden_dim, num_layers=2, curvature=hyperbolicity, dropout=0.6)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    return model


class Transformer_v2(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, heads=1, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                TransformerConv(hidden_channels, hidden_channels, heads, concat=False))
            
        self.dropout = dropout

    def reset_parameters(self):
        # super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.lins[0](x).relu()
        # x = F.dropout(x, self.dropout, training=self.training)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = x.relu()
            # x = F.dropout(x, self.dropout, training=self.training)

        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x.log_softmax(dim=-1)
    

class Transformer(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, heads=1, dropout=0.0):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(dataset.num_features, hidden_channels, heads, concat=True))
        for layer in range(num_layers-2):
            self.convs.append(
                TransformerConv(hidden_channels*heads, hidden_channels*heads, heads, concat=False))
        self.convs.append(TransformerConv(hidden_channels*heads, dataset.num_classes, heads, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        return F.log_softmax(x, dim=1)


# class Transformer(torch.nn.Module):
#     def __init__(self, dataset, hidden_channels, num_layers=2, heads=1, dropout=0.0):
#         super().__init__()

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(TransformerConv(dataset.num_features, hidden_channels, heads))
#         for layer in range(num_layers-2):
#             self.convs.append(
#                 TransformerConv(hidden_channels*heads, hidden_channels*heads, heads))
#         self.convs.append(TransformerConv(hidden_channels*heads, dataset.num_classes, heads=1))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
    
#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index, edge_attr)
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index, edge_attr)

#         return F.log_softmax(x, dim=1)




class SAGE(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(dataset.num_features, hidden_channels))
        for layer in range(num_layers-2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, dataset.num_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        return F.log_softmax(x, dim=1)




class Spline(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SplineConv(dataset.num_features, hidden_channels, dim=1, kernel_size=2))
        for layer in range(num_layers-2):
            self.convs.append(
                SplineConv(hidden_channels, hidden_channels, dim=1, kernel_size=2))
        self.convs.append(SplineConv(hidden_channels, dataset.num_classes, dim=1, kernel_size=2))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        return F.log_softmax(x, dim=1)

class GIN_v2(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()

        self.act = torch.nn.ReLU()
        self.act_first = True

        self.convs = torch.nn.ModuleList()
        mlp = MLP(
            [dataset.num_features, hidden_channels, dataset.num_classes],
            act=self.act,
            act_first=self.act_first,
            num_layers= 3)
        self.convs.append(GINConv(mlp, train_eps=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)

class GIN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()

        self.act = torch.nn.ReLU()
        self.act_first = True

        self.convs = torch.nn.ModuleList()
        mlp = MLP(
            [dataset.num_features, hidden_channels, hidden_channels],
            act=self.act,
            act_first=self.act_first,
            num_layers= 1)
        self.convs.append(GINConv(mlp, train_eps=True))
        for layer in range(num_layers-2):
            mlp = MLP(
                [hidden_channels, hidden_channels, hidden_channels],
                act=self.act,
                act_first=self.act_first,
                num_layers= 1)
            self.convs.append(
                GINConv(mlp))
            
        mlp = MLP(
            [hidden_channels, hidden_channels, dataset.num_classes],
            act=self.act,
            act_first=self.act_first,
            num_layers= 1)
        self.convs.append(GINConv(mlp, train_eps=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)


class GAT_v2(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, heads=1, dropout=0.0):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels, heads, concat=True))
        for layer in range(num_layers-2):
            self.convs.append(
                GATConv(hidden_channels*heads, hidden_channels*heads, heads, concat=False))
        self.convs.append(GATConv(hidden_channels*heads, dataset.num_classes, heads, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)
    

class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, heads=1, dropout=0.0):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels, heads, normalize=False, cached=False))
        for layer in range(num_layers-2):
            self.convs.append(
                GATConv(hidden_channels*heads, hidden_channels*heads, heads, normalize=False, cached=False))
        self.convs.append(GATConv(hidden_channels*heads, dataset.num_classes, heads=1, normalize=False, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)



class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_features, hidden_channels, normalize=True, cached=True))
        for layer in range(num_layers-2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels,  normalize=True, cached=True))
        self.convs.append(GCNConv(hidden_channels, dataset.num_classes, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)
    


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
    
class GCN_SSL(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, dropout):
        super(GCN_SSL, self).__init__()
        self.crd = CRD(dataset.num_features, hidden_dim, dropout)
        self.cls = CLS(hidden_dim, dataset.num_classes)

    def reset_parameters(self):
        # super().reset_parameters()
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x


# class wrapperGCN(GCN):
#     def __init__(self, dataset, hidden_channels, num_layers, dropout) -> None:
#         super().__init__(in_channels=dataset.num_features, 
#                          hidden_channels=hidden_channels, 
#                          out_channels=dataset.num_classes, 
#                          num_layers=num_layers,
#                          dropout=dropout)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         return super(wrapperGCN, self).forward(x, edge_index)
    

# class GCN(torch.nn.Module):
#     def __init__(self, dataset, hidden_channels, num_layers, dropout=0.0):
#         super().__init__()

#         self.lins = torch.nn.ModuleList()
#         self.lins.append(Linear(dataset.num_features, hidden_channels))
#         self.lins.append(Linear(hidden_channels, dataset.num_classes))

#         self.convs = torch.nn.ModuleList()
#         for layer in range(num_layers):
#             self.convs.append(
#                 GCNConv(hidden_channels, hidden_channels,  normalize=False))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for lin in self.lins:
#             lin.reset_parameters()

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.lins[0](x).relu()

#         for conv in self.convs:
#             x = F.dropout(x, self.dropout, training=self.training)
#             x = conv(x, edge_index)
#             x = x.relu()

#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.lins[1](x)

#         return x.log_softmax(dim=-1)
    




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
        # super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

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
    

class GCN2_BP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0, manifold='PoincareBall'):
        super().__init__()

        self.manifold = getattr(manifolds, manifold)()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels**2, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
    
    def outer_product(self, x):              
        # Expand the dimensions of x to (batch, feats, 1)
        x = x.unsqueeze(2)
        
        # Compute the outer product using torch.matmul
        result = torch.matmul(x, x.transpose(1, 2))
        
        return result

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        h = h_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(h, self.dropout, training=self.training)
            h = conv(h, h_0, adj_t)
            h = h.relu()

        # Outter product
        h = self.outer_product(h)
        h = torch.flatten(h, start_dim=1)

        h = F.dropout(h, self.dropout, training=self.training)
        h = self.lins[1](h)

        return h.log_softmax(dim=-1)

class GCN2_HBP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers, alpha, theta, curvature,
                 shared_weights=True, dropout=0.0, manifold='PoincareBall'):
        super().__init__()

        self.manifold = getattr(manifolds, manifold)()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels**2, dataset.num_classes))
        self.c = curvature

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()


    # def hyperbolic_projection(self, x):
    #     # Compute the norm of each row in x
    #     norm_x = torch.norm(x, dim=1, keepdim=True)
        
    #     # Compute the sinh and cosh of the norm
    #     sinh_norm_x = torch.sinh(norm_x)
    #     cosh_norm_x = torch.cosh(norm_x)
        
    #     # Compute the hyperbolic projection
    #     projected_x = sinh_norm_x * (x / norm_x)
        
    #     return projected_x
    
    def outer_product(self, x):              
        # Expand the dimensions of x to (batch, feats, 1)
        x = x.unsqueeze(2)
        
        # Compute the outer product using torch.matmul
        result = torch.matmul(x, x.transpose(1, 2))
        
        return result

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        h = h_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(h, self.dropout, training=self.training)
            h = conv(h, h_0, adj_t)
            h = h.relu()

        # Outter product
        h_hyp_o = self.outer_product(h)
        h_hyp_o = torch.flatten(h_hyp_o, start_dim=1)

        # Hyperbolic -> euclidean
        # x = self.hyperbolic_projection(x)
        h_hyp = self.manifold.proj(h_hyp_o, c=self.c)       # Normalize to the hyperboloid
        h_euc = self.manifold.proj_tan0(self.manifold.logmap0(h_hyp, c=self.c), c=self.c)

        h_euc = F.dropout(h_euc, self.dropout, training=self.training)
        h_euc = self.lins[1](h_euc)

        # Euclidean -> hyperbolic
        h_tan = self.manifold.proj_tan0(h_euc, self.c)
        h_hyp = self.manifold.expmap0(h_tan, c=self.c)
        h_hyp = self.manifold.proj(h_hyp, c=self.c)

        # x = torch.einsum('imjk,injk->imn', x, y)  # [b,m,n]

        return h_hyp.log_softmax(dim=-1)
    

class GCN_HBP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, curvature=0, num_layers=2, dropout=0.0, manifold='PoincareBall'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_features, hidden_channels, normalize=True, cached=True))
        for layer in range(num_layers-2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels,  normalize=True, cached=True))
        self.convs.append(GCNConv(hidden_channels, dataset.num_classes, normalize=True, cached=True))

        self.manifold = getattr(manifolds, manifold)()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(dataset.num_classes**2, dataset.num_classes))
        self.c = curvature
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def outer_product(self, x):              
        # Expand the dimensions of x to (batch, feats, 1)
        x = x.unsqueeze(2)
        
        # Compute the outer product using torch.matmul
        result = torch.matmul(x, x.transpose(1, 2))
        
        return result
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        # Outter product
        h_hyp_o = self.outer_product(x)
        h_hyp_o = torch.flatten(h_hyp_o, start_dim=1)

        # Hyperbolic -> euclidean
        # x = self.hyperbolic_projection(x)
        h_hyp = self.manifold.proj(h_hyp_o, c=self.c)       # Normalize to the hyperboloid
        h_euc = self.manifold.proj_tan0(self.manifold.logmap0(h_hyp, c=self.c), c=self.c)

        h_euc = F.dropout(h_euc, self.dropout, training=self.training)
        h_euc = self.lins[1](h_euc)

        # Euclidean -> hyperbolic
        h_tan = self.manifold.proj_tan0(h_euc, self.c)
        h_hyp = self.manifold.expmap0(h_tan, c=self.c)
        h_hyp = self.manifold.proj(h_hyp, c=self.c)

        # x = torch.einsum('imjk,injk->imn', x, y)  # [b,m,n]

        return h_hyp.log_softmax(dim=-1)


class SAGE_HBP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, curvature=0, num_layers=2, dropout=0.0, manifold='PoincareBall'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(dataset.num_features, hidden_channels))
        for layer in range(num_layers-2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, dataset.num_classes))
        self.manifold = getattr(manifolds, manifold)()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(dataset.num_classes**2, dataset.num_classes))
        self.c = curvature
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
    def outer_product(self, x):              
        # Expand the dimensions of x to (batch, feats, 1)
        x = x.unsqueeze(2)
        
        # Compute the outer product using torch.matmul
        result = torch.matmul(x, x.transpose(1, 2))
        
        return result
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        # Outter product
        h_hyp_o = self.outer_product(x)
        h_hyp_o = torch.flatten(h_hyp_o, start_dim=1)

        # Hyperbolic -> euclidean
        # x = self.hyperbolic_projection(x)
        h_hyp = self.manifold.proj(h_hyp_o, c=self.c)       # Normalize to the hyperboloid
        h_euc = self.manifold.proj_tan0(self.manifold.logmap0(h_hyp, c=self.c), c=self.c)

        h_euc = F.dropout(h_euc, self.dropout, training=self.training)
        h_euc = self.lins[1](h_euc)

        # Euclidean -> hyperbolic
        h_tan = self.manifold.proj_tan0(h_euc, self.c)
        h_hyp = self.manifold.expmap0(h_tan, c=self.c)
        h_hyp = self.manifold.proj(h_hyp, c=self.c)

        # x = torch.einsum('imjk,injk->imn', x, y)  # [b,m,n]

        return h_hyp.log_softmax(dim=-1)
    


class GCN2_HBPU(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers, alpha, theta, curvature,
                 shared_weights=True, dropout=0.0, manifold='PoincareBall'):
        super().__init__()

        self.manifold = getattr(manifolds, manifold)()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels**2, 64))
        self.lins.append(Linear(64, dataset.num_classes))
        self.c = curvature

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    
    def outer_product(self, x):              
        # Expand the dimensions of x to (batch, feats, 1)
        x = x.unsqueeze(2)
        
        # Compute the outer product using torch.matmul
        result = torch.matmul(x, x.transpose(1, 2))
        
        return result


    def regularizer(self, args):

        if args.regularization == 'uniformity':
            weights = self.lins[2].weight
            num_classes = weights.shape[0]

            # Euclidean -> hyperbolic
            w_tan = self.manifold.proj_tan0(weights, self.c)
            w_hyp = self.manifold.expmap0(w_tan, c=self.c)
            w_hyp = self.manifold.proj(w_hyp, c=self.c)

            # Create a zeros matrix of size (num_classes, num_classes)
            d = torch.zeros((num_classes, num_classes))

            for i, p1 in enumerate(w_hyp):
                for j, p2 in enumerate(w_hyp):
                    d[i,j] = self.manifold.sqdist(p1, p2, self.c)
    
            # print(d)

            # Approximate the minumum distance between the points (avoiding self-loops)
            d =  d + torch.eye(num_classes) * torch.max(d)
            reg = temperature_scaled_logminexp(d, tau=0.1)
        else:
            raise NotImplementedError(f"Regularization {args.regularization} not implemented.")
        
        # Check if contains nan 
        if torch.isnan(reg) or torch.isinf(reg):
            raise ValueError("Regularization is NaN or Inf.")

        return reg
    
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        h = h_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(h, self.dropout, training=self.training)
            h = conv(h, h_0, adj_t)
            h = h.relu()

        # Outter product
        h_hyp_o = self.outer_product(h)
        h_hyp_o = torch.flatten(h_hyp_o, start_dim=1)

        # Hyperbolic -> euclidean
        h_hyp = self.manifold.proj(h_hyp_o, c=self.c)       # Normalize to the hyperboloid
        h_euc = self.manifold.proj_tan0(self.manifold.logmap0(h_hyp, c=self.c), c=self.c)

        h_euc = F.dropout(h_euc, self.dropout, training=self.training)
        h_euc = self.lins[1](h_euc)


        h_euc = F.dropout(h_euc, self.dropout, training=self.training)
        h_euc = self.lins[2](h_euc)


        # Euclidean -> hyperbolic
        h_tan = self.manifold.proj_tan0(h_euc, self.c)
        h_hyp = self.manifold.expmap0(h_tan, c=self.c)
        h_hyp = self.manifold.proj(h_hyp, c=self.c)

        return h_hyp.log_softmax(dim=-1)


# Low Rank version of BP
class GCN2_HLBP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers, alpha, theta, curvature,
                 shared_weights=True, dropout=0.0, manifold='PoincareBall'):
        super().__init__()

        # Dimension of the low-rank
        self.low_dim = int(hidden_channels/1)

        self.manifold = getattr(manifolds, manifold)()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(self.low_dim**2, dataset.num_classes))
        self.c = curvature

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

        self.r = 3
        for i in range(self.r):
            setattr(self, f"u{i}", Linear(hidden_channels, self.low_dim, bias=True))
            setattr(self, f"v{i}", Linear(hidden_channels, self.low_dim, bias=True))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        h = h_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(h, self.dropout, training=self.training)
            h = conv(h, h_0, adj_t)
            h = h.relu()

        # Hyperbolic -> euclidean
        h_hyp = self.manifold.proj(h, c=self.c)       # Normalize to the hyperboloid
        h_euc = self.manifold.proj_tan0(self.manifold.logmap0(h_hyp, c=self.c), c=self.c)

        # Low-ranked
        h_bp = torch.zeros((h.shape[0], self.low_dim, self.low_dim)).to(h.device)
        for i in range(self.r):
            h_u = getattr(self, f"u{i}")(h).unsqueeze(-1)
            h_v = getattr(self, f"v{i}")(h).unsqueeze(-1)
            h_bp += torch.einsum('imk,inl->imnl', h_u, h_v).squeeze(-1)
            
        h_flat = torch.flatten(h_bp, start_dim=1)
        h_euc = h_flat

        # Hyperbolic -> euclidean
        # h_hyp = self.manifold.proj(h_flat, c=self.c)       # Normalize to the hyperboloid
        # h_euc = self.manifold.proj_tan0(self.manifold.logmap0(h_hyp, c=self.c), c=self.c)

        h_euc = F.dropout(h_euc, self.dropout, training=self.training)
        h_euc = self.lins[1](h_euc)

        # Euclidean -> hyperbolic
        h_tan = self.manifold.proj_tan0(h_euc, self.c)
        h_hyp = self.manifold.expmap0(h_tan, c=self.c)
        h_hyp = self.manifold.proj(h_hyp, c=self.c)

        return h_hyp.log_softmax(dim=-1)