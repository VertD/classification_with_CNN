import os
import json

import pandas as pd
import torch
from torch.utils.data import Dataset


class PatientDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        """
        annotations_file: путь к JSON-файлу с разметкой
        transform: любые преобразования, которые вы хотите применить к тензору
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.transform = transform
        self.root_dir = os.path.dirname(annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        patient_info = self.annotations[idx]

        tensor_path = patient_info['tensor']
        label = patient_info['class']

        # Загружаем тензор
        full_tensor_path = os.path.join(self.root_dir, tensor_path)
        tensor = torch.load(full_tensor_path)

        # Применяем трансформации (если есть)
        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label
