import torch as th
import torch.nn as nn
from sklearn.metrics import accuracy_score

import numpy as np
import os
import time

def eval(model, device, eval_loader):
    labels_true = []
    labels_pred = []
    for step, (data, label) in enumerate(eval_loader, start=1):
        data = data.to(device)

        with th.no_grad():
            label_pred = model(data).argmax(axis=-1).squeeze()

        labels_true.append(label.numpy())
        labels_pred.append(label_pred.cpu().numpy())
        
    score = accuracy_score(np.concatenate(labels_true), np.concatenate(labels_pred))
    return score

def train(model, device, path, run, train_loader, eval_loader, num_epoch):
    model.to(device)
    run.watch(model)

    optimizer = th.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    epoch_stat = {"epoch": 0}

    score = eval(model, device, eval_loader)
    epoch_stat["eval/accuracy"] = score

    min_epoch_score = score
    th.save(model.state_dict(), os.path.join(path, f"best_model.pt"))

    run.log(epoch_stat)

    cumulative_time = 0

    for epoch in range(1, num_epoch + 1):
        model.train()
        loss_accum = 0
        loss_count = 0

        epoch_stat = {"epoch": epoch}

        start_train_time = time.time()
        for step, (data, label) in enumerate(train_loader, start=1):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            label_logits = model(data)
            loss = criterion(label_logits, label)
            loss.backward()
            optimizer.step()

            loss_accum += loss.item() * data.shape[0]
            loss_count += data.shape[0]
        end_train_time = time.time()
            
        epoch_stat["train/loss"] = loss_accum / loss_count
        epoch_stat["train/time"] = end_train_time - start_train_time

        cumulative_time += end_train_time - start_train_time
        epoch_stat["train/cumulative_time"] = cumulative_time

        model.eval()
        
        score = eval(model, device, eval_loader)
        epoch_stat["eval/accuracy"] = score

        run.log(epoch_stat)

        if score > min_epoch_score:
            min_epoch_score = score
            th.save(model.state_dict(), os.path.join(path, f"best_model.pt"))