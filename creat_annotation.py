import os
import json

def create_annotation(root_dir, output_file, class_map):
    """
    root_dir: str — путь к корневой папке (train или test)
    output_file: str — имя выходного JSON-файла
    class_map: dict — сопоставление подпапок (healthy/sick) с метками классов
    """

    annotations = []

    for class_name, label in class_map.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            if filename.endswith(".png"):
                patient_name = os.path.splitext(filename)[0]
                # Добавляем полный путь относительно root_dir
                image_path = os.path.join(root_dir, class_name, filename)
                annotations.append({
                    "patient": patient_name,
                    "class": label,
                    "image": image_path  # полный путь относительно корня
                })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    class_labels = {
        "healthy": 0,
        "sick": 1
    }
    create_annotation("train", "train_annotations.json", class_labels)
    create_annotation("test", "test_annotations.json", class_labels)
