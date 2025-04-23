import joblib
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
import torch
from scipy.interpolate import interp1d

# Создаём все нужные директории
base_dirs = ['train/healthy', 'train/sick', 'test/healthy', 'test/sick']
for dir_path in base_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Загружаем данные
data = joblib.load('data_raw.joblib')

# Фамилии по группам
first_group_part1 = [
    'Абдуллаева', 'Александров', 'Буй', 'Василин', 'Вековшинин', 'Глебов',
    'Головков', 'Горев', 'Горягин', 'Звягина', 'Киселев', 'Косолапов',
    'Костицина', 'Липатова', 'Лямина', 'Мальсачов', 'Меруовский',
    'Михайлов', 'Назарова', 'Полякова', 'Рагрин', 'Скородумов', 'Стайко',
    'Шантуров', 'Шарифуллин'
]

second_group_part1 = [
    'Юань', 'Ахматов', 'Бажин', 'Белов', 'Белоусов', 'Буйняков',
    'Воронцов', 'Грязнов', 'Джумбиев', 'Дзядковский', 'Иванов',
    'Карельский', 'Коваленко', 'Колесник', 'Котов', 'Кочерин',
    'Куманин', 'Липатов', 'Маляревский', 'Машурчак', 'Менделеев',
    'Новоженин', 'Пащенко', 'Петров', 'Попов', 'Самедов', 'Сидибе',
    'Симонов', 'Скачков', 'Соловьёв', 'Соловьёва', 'Тульчинский',
    'Хазимов', 'Хасбулатов', 'Чертков', 'Шурпатов'
]

# Формируем индексы
first = ceil(len(first_group_part1) * 0.7)
second = ceil(len(second_group_part1) * 0.7)

train_names = first_group_part1[:first] + second_group_part1[:second]
test_names = [idx for idx in data.index if idx not in train_names]

train_df = data.loc[train_names]
test_df = data.loc[test_names]

healthy_train_names = set(first_group_part1[:first])
healthy_test_names = set(first_group_part1[first:])
sick_train_names = set(second_group_part1[:second])
sick_test_names = set(second_group_part1[second:])

# Нормализация
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-8) if max_val != min_val else np.zeros_like(data)

# Возвращает x и усреднённые y
def pad_or_trim(data):
    tensor = torch.empty(100, 17)  # 100 строк, 17 столбцов
    for i, col in enumerate(data.columns):
        y = data[col].values
        x_old = np.linspace(0, 1, len(y))            # оригинальные точки
        x_new = np.linspace(0, 1, 100)               # новые точки для интерполяции
        f = interp1d(x_old, y, kind='linear', fill_value="extrapolate")
        y_new = f(x_new)
        tensor[:, i] = torch.tensor(normalize_data(y_new), dtype=torch.float32)  # Транспонируем данные
    df = pd.DataFrame(tensor)
    print(df)
    return tensor
# Построение графиков
def make_group(df, healthy_set, sick_set, root_dir):
    for idx, value in df['data'].items():
        first_entry = value
        condition_name = list(first_entry.keys())[0]
        table_raw = first_entry[condition_name]
        first_key = list(table_raw.keys())[0]
        dat = table_raw[first_key].iloc[:, 4:].drop(table_raw[first_key].columns[9], axis=1)

        tensor = pad_or_trim(dat)

        if idx in healthy_set:
            torch.save(tensor, os.path.join(root_dir, 'healthy', f'{idx}.pt'))
        elif idx in sick_set:
            torch.save(tensor, os.path.join(root_dir, 'sick', f'{idx}.pt'))

# Генерируем графики
make_group(train_df, healthy_train_names, sick_train_names, root_dir='train')
make_group(test_df, healthy_test_names, sick_test_names, root_dir='test')
