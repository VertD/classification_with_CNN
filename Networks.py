import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import PatientDataset

class Conv2DClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.ConvLayer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 100x17 → 50x8

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50x8 → 25x4

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25x4 → 12x2
        )

        # Полносвязные слои
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
        return x  # Вероятность класса sick (healthy = 1 - sick)


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



batch_size = 2
epochs = 12
learning_rate = 0.001

# Создаем датасет и DataLoader

train_dataset = PatientDataset('train_annotations.json')
test_dataset = PatientDataset('test_annotations.json')


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# Модель, оптимизатор и функция потерь
model = Conv1DClassifier()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_func = nn.BCEWithLogitsLoss()


# Функция для обучения модели
train_losses = []  # Список для хранения значений loss
def train(model, train_loader, loss_func, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            #images = images.unsqueeze(1) для 2d
            images = images.permute(0, 2, 1) #для 1d
            logits = model(images).squeeze(1)
            loss = loss_func(logits, labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Переводим вероятность в класс (sick = 1)
            predicted = (logits > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)  # Сохраняем loss для этой эпохи
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


# Тест
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            #images = images.unsqueeze(1) для 2d
            images = images.permute(0, 2, 1) # для 1d
            logits = model(images).squeeze(1)
            predicted = (logits > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

# Обучаем модель
train(model, train_loader, loss_func, optimizer, epochs)

# Оцениваем модель на тестовых данных
evaluate(model, test_loader)

torch.save(model.state_dict(), "2d_cnn_best_params.pth")

plt.plot(range(epochs), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.legend()
plt.show()


