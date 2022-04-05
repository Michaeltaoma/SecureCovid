import copy
import time

import numpy as np
import torch
import tqdm.notebook as tqdm
from sklearn.metrics import cohen_kappa_score
from torch.nn.utils import clip_grad_norm_


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_attack_model(device, model, criterion, optimizer, dataloaders, num_epochs=10):
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    for e in tqdm.tqdm(range(num_epochs)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch.float()).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = binary_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch.float()).squeeze()
                # y_val_pred = torch.unsqueeze(y_val_pred, 0)
                # y_val_batch = torch.unsqueeze(y_val_batch, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = binary_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))
        if e % 2000 == 0:
            print(
                f'Epoch {e + 0:02}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}%| Val Acc: {val_epoch_acc / len(val_loader):.3f}%')

    return model, loss_stats, accuracy_stats


def train_model(device, model, criterion, optimizer, scheduler, data_sizes, dataloaders, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    epoch_loss_record = list()
    epoch_acc_record = list()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            current_loss = 0.0
            current_corrects = 0
            current_kappa = 0
            val_kappa = list()

            for inputs, labels in tqdm.tqdm(dataloaders[phase], desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if phase == 'train':
                    scheduler.step()

                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)
                val_kappa.append(cohen_kappa_score(preds.cpu().numpy(), labels.data.cpu().numpy()))
            epoch_loss = current_loss / data_sizes[phase]
            epoch_acc = current_corrects.double() / data_sizes[phase]
            epoch_loss_record.append(epoch_loss.cpu())
            epoch_acc_record.append(epoch_acc.cpu())
            if phase == 'val':
                epoch_kappa = np.mean(val_kappa)
                print('{} Loss: {:.4f} | {} Accuracy: {:.4f} | Kappa Score: {:.4f}'.format(
                    phase, epoch_loss, phase, epoch_acc, epoch_kappa))
            else:
                print('{} Loss: {:.4f} | {} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, phase, epoch_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('Val loss Decreased from {:.4f} to {:.4f} \nSaving Weights... '.format(best_loss, epoch_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val loss: {:.4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)

    return model, epoch_loss_record, epoch_acc_record


def train_model_with_dp(device, model, criterion, dataloaders, num_epochs=10, noise_multiplier=1.0, max_grad_norm=1.2, lr=0.02):
    for epoch in range(num_epochs):
        epoch_acc = []
        for batch in dataloaders["train"]:
            batch_len = batch[1].shape[0]
            acc = 0
            for param in model.parameters():
                param.accumulated_grads = []

            # Run the microbatches
            for x, y in zip(batch[0], batch[1]):
                x = torch.unsqueeze(x, 0).to(device)
                y = torch.unsqueeze(y, 0).to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                _, preds = torch.max(y_hat, 1)
                acc += 1 if preds.data == y.data else 0
                loss.backward()

                # Clip each parameter's per-sample gradient
                for param in model.parameters():
                    per_sample_grad = param.grad.detach().clone()
                    clip_grad_norm_(per_sample_grad, max_norm=max_grad_norm)  # in-place
                    param.accumulated_grads.append(per_sample_grad)

            # Aggregate back
            for param in model.parameters():
                # a = torch.sum(torch.stack(param.accumulated_grads, dim=0), dim=0)
                param.grad = torch.sum(torch.stack(param.accumulated_grads, dim=0), dim=0)

            # Now we are ready to update and add noise!
            for param in model.parameters():
                param = param - lr * param.grad
                mean = torch.zeros(param.shape)
                std = torch.full(param.shape, noise_multiplier * max_grad_norm)
                param += torch.normal(mean=mean, std=std).to(device)

                param.grad = torch.zeros(param.shape)  # Reset for next iteration

            epoch_acc.append(acc / batch_len)

            if (len(epoch_acc) % 10 == 0):
                print("Batch Acc is {:.2f}".format(sum(epoch_acc) / len(epoch_acc)))

        print("Epoch Acc is {:.2f}".format(sum(epoch_acc) / len(epoch_acc)))

    return model
