import os.path as osp
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T




def get_planetoid_dataset(model_name, dataset, normalize_features=False, transform=None, split="public"):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', dataset)
    if split == 'complete':
        dataset = Planetoid(path, dataset)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path, dataset, split=split)


    # Set PyG Transform according to model
    if 'GCN2' in model_name:
        transform = T.Compose([#T.NormalizeFeatures(),   # Normalize data.x
                            T.GCNNorm(),            # Create data.edge_weight
                            T.ToSparseTensor()])    # Store edge_weight as sparse -> adj_t
    elif 'Spline' in model_name:
        transform = T.Compose([T.TargetIndegree()])
    elif 'Transformer' in model_name:
        transform = T.Constant(node_types='conference')
    
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset





if __name__ == '__main__':
    lst_names = ['Cora', 'CiteSeer', 'PubMed']
    for name in lst_names:
        dataset = get_planetoid_dataset(name)
        print(f"dataset: {name}")
        print(f"num_nodes: {dataset[0]['x'].shape[0]}")
        print(f"num_edges: {dataset[0]['edge_index'].shape[1]}")
        print(f"num_classes: {dataset.num_classes}")
        print(f"num_features: {dataset.num_node_features}")