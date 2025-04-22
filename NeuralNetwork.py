import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import PatientDataset

class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.ConvLayer = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 180x245 → 90x122

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 90x122 → 45x61

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),  # 45x61 → 11x15
        )

        # Полносвязные слои
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 11 * 15, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.ConvLayer(x)
        x = self.out(x)
        return x  # Вероятность класса sick (healthy = 1 - sick)


batch_size = 8
epochs = 16
learning_rate = 0.001

# Создаем датасет и DataLoader
transform = transforms.Compose([
    transforms.CenterCrop((360, 490)),
    transforms.Resize((180, 245)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_dataset = PatientDataset('train_annotations.json', transform=transform)
test_dataset = PatientDataset('test_annotations.json', transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# Модель, оптимизатор и функция потерь
model = CNN_Model()
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

torch.save(model.state_dict(), "cnn_model_test.pth")

plt.plot(range(epochs), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.legend()
plt.show()


saved_model = CNN_Model()
saved_model.load_state_dict(torch.load("cnn_model_final.pth"))
saved_model.eval()
print('весь датасет на лучшей моделе')
evaluate(saved_model, full_loader)

saved_model2 = CNN_Model()
saved_model2.load_state_dict(torch.load("cnn_model_test.pth"))
saved_model.eval()
print('весь датасет на текующей моделе')
evaluate(saved_model2, full_loader)