import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import PatientDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import seaborn as sns

class Conv2DClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.ConvLayer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(11, 11), padding=(5, 5)),  # широкий фильтр
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),  # чуть уже
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),  # мелкий фильтр
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.ConvLayer(x)
        x = self.out(x)
        return x

class Conv1DClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(17, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 → 50

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 50 → 25

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.fc(x)

class RNNClassifier(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,      # 17 признаков на каждом шаге
            hidden_size=hidden_size,    # размер скрытого состояния
            num_layers=num_layers,
            batch_first=True,           # вход: (batch, seq_len, input_size)
            dropout=dropout,
            nonlinearity='tanh'         # можно 'tanh' или 'relu'
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)            # бинарная классификация
        )

    def forward(self, x):
        # x: (batch_size, seq_len=100, input_size=17)
        rnn_out, h_n = self.rnn(x)
        # h_n: (num_layers, batch, hidden_size)
        final_hidden_state = h_n[-1]  # берём последний слой
        out = self.classifier(final_hidden_state)
        return out


batch_size = 2
epochs = 20
learning_rate = 0.001

# Создаем датасет и DataLoader

train_dataset = PatientDataset('train_annotations.json')
test_dataset = PatientDataset('test_annotations.json')


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Модель, оптимизатор и функция потерь
model = Conv2DClassifier()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_func = nn.BCEWithLogitsLoss()


# Функция для обучения модели
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def train(model, train_loader, val_loader, loss_func, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            #images = images # для rnn
            images = images.unsqueeze(1) #для 2d
            #images = images.permute(0, 2, 1) #для 1d
            logits = model(images).squeeze(1)
            loss = loss_func(logits, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (logits > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # валидация
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.unsqueeze(1) #для 2d
                #images = images  # для rnn
                #images = images.permute(0, 2, 1)  # для 1d
                logits = model(images).squeeze(1)
                val_loss = loss_func(logits, labels.float())
                val_running_loss += val_loss.item()

                predicted = (logits > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")


# --- Отрисовка графиков ---
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.xticks(epochs)
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_with_metrics(model, dataloader):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.unsqueeze(1) #для 2d
            #images = images  # для rnn
            #images = images.permute(0, 2, 1)  # если Conv1DClassifier
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    # Confusion Matrix по порогу 0.5
    predicted = [1 if p > 0.5 else 0 for p in all_probs]
    cm = confusion_matrix(all_labels, predicted)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Визуализация confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC кривая
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()



train(model, train_loader, val_loader, loss_func, optimizer, epochs)
# torch.save(model.state_dict(), "2d_cnn_best_params.pth")
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
evaluate_with_metrics(model, val_loader)