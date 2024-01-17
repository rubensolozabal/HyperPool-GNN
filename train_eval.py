from __future__ import division

import time
import os
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import utils as ut
from tqdm import tqdm

# Macro definition
PATH_RUNS = "runs"

def run(
    dataset, 
    model, 
    str_optimizer, 
    runs, 
    epochs, 
    lr, 
    weight_decay, 
    early_stopping,  
    logger, 
    momentum,
    eps,
    update_freq,
    hyperparam,
    device
    ):
    if logger is not None:
        if hyperparam:
            logger += f"-{hyperparam}{eval(hyperparam)}"
        path_logger = os.path.join(PATH_RUNS, logger)
        print(f"Path logger: {path_logger}")

        ut.empty_dir(path_logger)
        logger = SummaryWriter(log_dir=os.path.join(PATH_RUNS, logger)) if logger is not None else None

    val_losses, accs, durations = [], [], []

    for i_run in range(runs):
        data = dataset[0]
        data = data.to(device)

        model.to(device).reset_parameters()

        if str_optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif str_optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=momentum,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        # Training loop...
        with tqdm(range(1, epochs + 1)) as t:
            for epoch in t:
                t.set_description(f'Run: {i_run+1}')

                train(model, optimizer, data)
                eval_info = evaluate(model, data)
                eval_info['epoch'] = int(epoch)
                eval_info['run'] = int(i_run+1)
                eval_info['time'] = time.perf_counter() - t_start
                eval_info['eps'] = eps
                eval_info['update-freq'] = update_freq

                if logger is not None:
                    for k, v in eval_info.items():
                        logger.add_scalar(k, v, global_step=epoch)
                    
                        
                if eval_info['val loss'] < best_val_loss:
                    best_val_loss = eval_info['val loss']
                    test_acc = eval_info['test acc']

                val_loss_history.append(eval_info['val loss'])
                if early_stopping > 0 and epoch > epochs // 2:
                    tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                    if eval_info['val loss'] > tmp.mean().item():
                        break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
    
    if logger is not None:
        logger.close()
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print('Val Loss: {:.4f}, Test Accuracy: {:.2f} ± {:.2f}, Duration: {:.3f} \n'.
          format(loss.mean().item(),
                 100*acc.mean().item(),
                 100*acc.std().item(),
                 duration.mean().item()))

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False
    
    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    
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
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc

    return outs
