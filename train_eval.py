from __future__ import division

import time
import os
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import utils.general_utils as ut
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import recall_score

# Macro definition
PATH_RUNS = "runs"

def run(
    dataset, 
    model, 
    args,
    device
    ):
    if args.logger is not None:
        if args.hyperparam:
            logger += f"-{args.hyperparam}{eval(args.hyperparam)}"
        path_logger = os.path.join(PATH_RUNS, args.logger)
        print(f"Path logger: {path_logger}")

        ut.empty_dir(path_logger)
        logger = SummaryWriter(log_dir=os.path.join(PATH_RUNS, args.logger)) if args.logger is not None else None

    val_losses, accs, durations = [], [], []
    f1s, aucs, nmis, aris, recall_1s, recall_2s = [], [], [], [], [], []

    for i_run in range(args.runs):
        data = dataset[0]
        data = data.to(device)

        model = model.to(device)
        # model.reset_parameters()

        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )

            if 'GCN2' in args.model:
                optimizer = torch.optim.Adam([
                dict(params=model.convs.parameters(), weight_decay=0.01),
                dict(params=model.lins.parameters(), weight_decay=5e-4)], lr=0.01
                )
            
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=args.lr, 
                momentum=args.momentum,
            )


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        # Training loop...
        with tqdm(range(1, args.epochs + 1)) as t:
            for epoch in t:
                t.set_description(f'Run: {i_run+1}')

                train(model, optimizer, data, args, device)
                eval_info = evaluate(model, data)
                eval_info['epoch'] = int(epoch)
                eval_info['run'] = int(i_run+1)
                eval_info['time'] = time.perf_counter() - t_start
                eval_info['eps'] = args.eps
                eval_info['update-freq'] = args.update_freq

                if args.logger is not None:
                    for k, v in eval_info.items():
                        logger.add_scalar(k, v, global_step=epoch)
                    
                        
                if eval_info['val loss'] < best_val_loss:
                    best_val_loss = eval_info['val loss']
                    test_acc = eval_info['test acc']
                    test_f1 = eval_info['test f1']
                    test_auc = eval_info['test auc']
                    test_nmi = eval_info['test nmi']
                    test_ari = eval_info['test ari']
                    test_recall_1 = eval_info['test recall_1']
                    test_recall_2 = eval_info['test recall_2']


                val_loss_history.append(eval_info['val loss'])
                if args.early_stopping > 0 and epoch > args.epochs // 2:
                    tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                    if eval_info['val loss'] > tmp.mean().item():
                        break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        f1s.append(test_f1)
        aucs.append(test_auc)
        nmis.append(test_nmi)
        aris.append(test_ari)
        recall_1s.append(test_recall_1)
        recall_2s.append(test_recall_2)
    
    if args.logger is not None:
        logger.close()
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    f1, auc, nmi, ari, recall_1, recall_2 = tensor(f1s), tensor(aucs), tensor(nmis), tensor(aris), tensor(recall_1s), tensor(recall_2s)


    results = {
        'loss': loss.mean().item(),
        'acc': acc.mean().item(),
        'loss_std': loss.std().item(),
        'acc_std': acc.std().item(),
        'duration': duration.mean().item(),
        'duration_std': duration.std().item(),
        'f1': f1.mean().item(),
        'f1_std': f1.std().item(),
        'auc': auc.mean().item(),
        'auc_std': auc.std().item(),
        'nmi': nmi.mean().item(),
        'nmi_std': nmi.std().item(),
        'ari': ari.mean().item(),
        'ari_std': ari.std().item(),
        'recall_1': recall_1.mean().item(),
        'recall_1_std': recall_1.std().item(),
        'recall_2': recall_2.mean().item(),
        'recall_2_std': recall_2.std().item(),
    }

    print('Val Loss: {:.4f}, Test Accuracy: {:.2f}±{:.2f}, Duration: {:.3f} \n'.
          format(loss.mean().item(),
                 100*acc.mean().item(),
                 100*acc.std().item(),
                 duration.mean().item()))
    
    return {**vars(args),**results}

def _get_loss(self, out, data, model, label, args, device):
    loss = F.nll_loss(out[data.train_mask], label[data.train_mask]) 
    
    # Adding loss regularization
    if args.regularization:
        loss = loss + (args.alpha * model.regularizer(args).to(device))

    return loss

def train(model, optimizer, data, args, device):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False
    
    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    # print("loss", loss)
    
    # Adding loss regularization
    if args.regularization:
        reg = args.alpha * model.regularizer(args).to(device)
        loss = loss - reg
        # print("reg", reg)
    
    loss.backward(retain_graph=True)
    optimizer.step()

def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()


        y_pred = logits[mask].max(1)[1].detach().cpu().numpy()
        y_true = data.y[mask].detach().cpu().numpy()
        logits_filtered = logits[mask].detach().cpu().numpy()
        y_prob = F.softmax(torch.tensor(logits_filtered), dim=1).numpy()

        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='macro')

        # Calculate AUC ROC
        auc = roc_auc_score(y_true, y_prob, average='macro',multi_class='ovo')

        # Calculate NMI
        nmi = normalized_mutual_info_score(y_true, y_pred)

        # Calculate ARI
        ari = adjusted_rand_score(y_true, y_pred)

        # Calculate Recall at 1 (R@1)
        recall_1 = recall_score(y_true, y_pred, average='micro')

        # Get the top 2 predicted classes for each instance
        top_2_classes = (-logits_filtered).argsort(axis=1)[:, :2]

        # Calculate Recall at 2 (R@2)
        recall_2 = 0
        for i in range(len(y_true)):
            if y_true[i] in top_2_classes[i]:
                recall_2 += 1
        recall_2 /= len(y_true)

        # # Calculate AUC ROC
        # logits_filtered = logits[mask].detach().cpu().numpy()
        # num_classes = logits_filtered.shape[1]
        # auc_roc_scores = []

        # for class_index in range(num_classes):
        #     y_true_class = np.where(y_true == class_index, 1, 0)
        #     y_pred_class = logits_filtered[:, class_index]

        #     auc_roc = roc_auc_score(y_true_class, y_pred_class)
        #     auc_roc_scores.append(auc_roc)

        # print(f"AUC ROC for {key} by class: {auc_roc_scores}")

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc
        outs['{} f1'.format(key)] = f1
        outs['{} auc'.format(key)] = auc
        outs['{} nmi'.format(key)] = nmi
        outs['{} ari'.format(key)] = ari
        outs['{} recall_1'.format(key)] = recall_1
        outs['{} recall_2'.format(key)] = recall_2

    return outs
