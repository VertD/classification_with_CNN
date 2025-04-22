import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PatientDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        """
        annotations_file: путь к JSON-файлу с разметкой
        transform: torchvision.transforms для графиков
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.transform = transform
        # Извлекаем корневую директорию изображений для корректного формирования пути
        self.root_dir = os.path.dirname(annotations_file)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        patient_info = self.annotations[idx]


        image_path = patient_info['image']

        label = patient_info['class']

        # Формируем полный путь к изображению
        full_image_path = os.path.join(self.root_dir, image_path)
        image = Image.open(full_image_path).convert("L")  # Ч/б изображение

        if self.transform:
            image = self.transform(image)

        return image, label

    def collate_fn(self, batch):
        # Убираем None элементы из батча
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None, None  # Если батч пустой, возвращаем None
        images, labels = zip(*batch)  # Разделяем изображения и метки
        return torch.stack(images, 0), torch.tensor(labels)
