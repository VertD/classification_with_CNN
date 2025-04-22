import joblib
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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
half_first = len(first_group_part1) // 2
half_second = len(second_group_part1) // 2

df1_index = first_group_part1[:half_first] + second_group_part1[:half_second]  # test
df2_index = [idx for idx in data.index if idx not in df1_index]               # train

test_df = data.loc[df1_index]
train_df = data.loc[df2_index]

healthy_test_names = set(first_group_part1[:half_first])
healthy_train_names = set(first_group_part1[half_first:])
sick_test_names = set(second_group_part1[:half_second])
sick_train_names = set(second_group_part1[half_second:])

# Нормализация
def normalize_data(data):
    data = data.astype(np.float32)
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-8) if max_val != min_val else np.zeros_like(data)

# Возвращает x и усреднённые y
def pad_or_trim(data):
    data = normalize_data(data)
    y = data.mean(axis=1)
    x = list(range(len(y)))
    return x, y

# Построение графиков
def make_group(df, healthy_set, sick_set, root_dir):
    for idx, value in df['data'].items():
        first_entry = value
        condition_name = list(first_entry.keys())[0]
        table_raw = first_entry[condition_name]
        first_key = list(table_raw.keys())[0]
        dat = table_raw[first_key].iloc[:, 4:].drop(table_raw[first_key].columns[9], axis=1)

        x, y = pad_or_trim(dat)
        plt.figure()
        plt.plot(x, y)

        if idx in healthy_set:
            plt.savefig(os.path.join(root_dir, 'healthy', f'{idx}.png'))
        elif idx in sick_set:
            plt.savefig(os.path.join(root_dir, 'sick', f'{idx}.png'))

        plt.close()

# Генерируем графики
make_group(train_df, healthy_train_names, sick_train_names, root_dir='train')
make_group(test_df, healthy_test_names, sick_test_names, root_dir='test')
