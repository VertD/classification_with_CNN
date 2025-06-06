import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PatientDataset
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    auc, confusion_matrix, accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Импорты моделей ---
from models import Conv1DClassifier, Conv2DClassifier, RNNClassifier  # предположим, они вынесены в models.py

# --- Настройка DataLoader ---
batch_size = 2
val_dataset = PatientDataset('test_annotations.json')  # валидационная выборка
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Оценка модели ---
def evaluate_model(model, loader, name, mode='rnn'):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, labels in loader:
            if mode == '1d':
                x = x.permute(0, 2, 1)
            elif mode == '2d':
                x = x.unsqueeze(1)

            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)

    # Метрики
    acc = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Графики
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} — ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(recall, precision, marker='.')
    plt.title(f"{name} — Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.tight_layout()
    plt.show()

# --- Загрузка моделей и параметров ---
models_info = [
    ("1D CNN", Conv1DClassifier(), "1d_cnn_best_params.pth", '1d'),
    ("2D CNN", Conv2DClassifier(), "2d_cnn_best_params.pth", '2d'),
    ("RNN", RNNClassifier(), "rnn_best_params.pth", 'rnn')
]

for name, model, path, mode in models_info:
    model.load_state_dict(torch.load(path))
    evaluate_model(model, val_loader, name, mode)
