import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PatientDataset
from models import Conv1DClassifier, Conv2DClassifier, RNNClassifier
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Параметры
epochs = 20
runs = 1
batch_size = 2
learning_rate = 0.0001

# Датасеты
train_dataset = PatientDataset('train_annotations.json')
val_dataset = PatientDataset('test_annotations.json')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def run_experiment(model_class, mode):
    train_losses_all = []
    val_losses_all = []
    train_accs_all = []
    val_accs_all = []

    for run in range(runs):
        model = model_class()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        loss_func = nn.BCEWithLogitsLoss()

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for x, y in train_loader:
                optimizer.zero_grad()

                if mode == '1d':
                    x = x.permute(0, 2, 1)
                elif mode == '2d':
                    x = x.unsqueeze(1)

                logits = model(x).squeeze(1)
                loss = loss_func(logits, y.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = (logits > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_losses.append(total_loss / len(train_loader))
            train_accs.append(100 * correct / total)

            # Валидация
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    if mode == '1d':
                        x = x.permute(0, 2, 1)
                    elif mode == '2d':
                        x = x.unsqueeze(1)

                    logits = model(x).squeeze(1)
                    loss = loss_func(logits, y.float())
                    val_loss += loss.item()

                    preds = (logits > 0.5).float()
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)

            val_losses.append(val_loss / len(val_loader))
            val_accs.append(100 * val_correct / val_total)

        train_losses_all.append(train_losses)
        val_losses_all.append(val_losses)
        train_accs_all.append(train_accs)
        val_accs_all.append(val_accs)

    # Усреднение
    return (
        np.mean(train_losses_all, axis=0),
        np.mean(val_losses_all, axis=0),
        np.mean(train_accs_all, axis=0),
        np.mean(val_accs_all, axis=0)
    )

# Запускаем эксперименты
results = {}
results['1D CNN'] = run_experiment(Conv1DClassifier, mode='1d')
results['2D CNN'] = run_experiment(Conv2DClassifier, mode='2d')
results['RNN'] = run_experiment(RNNClassifier, mode='rnn')

# Графики
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for name in results:
    plt.plot(epochs_range, results[name][3], label=f'{name}')
plt.title("Validation Accuracy (усреднённая)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.subplot(1, 2, 2)
for name in results:
    plt.plot(epochs_range, results[name][1], label=f'{name}')
plt.title("Validation Loss (усреднённая)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
