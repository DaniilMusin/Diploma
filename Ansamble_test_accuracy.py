import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Пути к весам моделей
model_paths = {
    "resnet50": "D:/Diploma/Binary/fine_tuned_acc_resnet50_fold2.pth",
    "densenet121": "D:/Diploma/Binary/fine_tuned_acc_densenet121_fold1.pth",
    "inceptionv3": "D:/Diploma/Binary/fine_tuned_acc_inceptionv3_fold2.pth", 
    "xception": "D:/Diploma/Binary/fine_tuned_acc_xception_fold3.pth"
}

class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.file_info = []
        self.img_dir = img_dir
        
        # Чтение CSV-файла
        with open(csv_file, 'r', encoding='utf-8') as f:
            next(f)  # Пропуск заголовка
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    label = int(parts[1].strip())
                    self.file_info.append((filename, label))
    
    def __len__(self):
        return len(self.file_info)
    
    def __getitem__(self, idx):
        img_name, label = self.file_info[idx]
        img_path = os.path.join(self.img_dir, img_name)
        return img_path, label

def load_models(device):
    models_dict = {}
    
    # ResNet50
    try:
        resnet = models.resnet50(weights=None)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, 2)
        resnet.load_state_dict(torch.load(model_paths["resnet50"], map_location=device))
        resnet.to(device)
        resnet.eval()
        models_dict["resnet50"] = resnet
        print("✓ ResNet50 загружен успешно")
    except Exception as e:
        print(f"✗ Ошибка загрузки ResNet50: {str(e)}")
    
    # DenseNet121
    try:
        densenet = models.densenet121(weights=None)
        in_features = densenet.classifier.in_features
        densenet.classifier = nn.Linear(in_features, 2)
        densenet.load_state_dict(torch.load(model_paths["densenet121"], map_location=device))
        densenet.to(device)
        densenet.eval()
        models_dict["densenet121"] = densenet
        print("✓ DenseNet121 загружен успешно")
    except Exception as e:
        print(f"✗ Ошибка загрузки DenseNet121: {str(e)}")
    
    # InceptionV3
    try:
        inception = models.inception_v3(weights=None, aux_logits=True)
        in_features = inception.fc.in_features
        inception.fc = nn.Linear(in_features, 2)
        aux_in_features = inception.AuxLogits.fc.in_features
        inception.AuxLogits.fc = nn.Linear(aux_in_features, 2)
        inception.load_state_dict(torch.load(model_paths["inceptionv3"], map_location=device))
        inception.to(device)
        inception.eval()
        models_dict["inceptionv3"] = inception
        print("✓ InceptionV3 загружен успешно")
    except Exception as e:
        print(f"✗ Ошибка загрузки InceptionV3: {str(e)}")
    
    # Xception
    try:
        xception = timm.create_model('legacy_xception', pretrained=False)
        in_features = xception.fc.in_features
        xception.fc = nn.Linear(in_features, 2)
        xception.load_state_dict(torch.load(model_paths["xception"], map_location=device))
        xception.to(device)
        xception.eval()
        models_dict["xception"] = xception
        print("✓ Xception загружен успешно")
    except Exception as e:
        print(f"✗ Ошибка загрузки Xception: {str(e)}")
    
    return models_dict

def get_transforms():
    transforms_dict = {}
    
    # ResNet50 и DenseNet121 (224x224)
    resnet_densenet_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # InceptionV3 и Xception (299x299)
    inception_xception_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transforms_dict["resnet50"] = resnet_densenet_transform
    transforms_dict["densenet121"] = resnet_densenet_transform
    transforms_dict["inceptionv3"] = inception_xception_transform
    transforms_dict["xception"] = inception_xception_transform
    
    return transforms_dict

def majority_vote(predictions):
    """Определяет итоговый класс голосованием большинства"""
    votes = np.bincount(predictions)
    return votes.argmax()

def plot_confusion_matrix(cm, class_names):
    """Отрисовка матрицы ошибок"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказано')
    plt.ylabel('Истинно')
    plt.title('Матрица ошибок')
    plt.savefig('confusion_matrix.png')
    plt.close()

def evaluate_ensemble():
    # Пути к тестовым данным
    csv_path = "D:/Diploma/labels_Denis.csv"
    img_dir = "D:/Diploma/Plates/Processed_Grayscale_Images"
    
    # Загрузка тестового набора
    print("\n1. Загрузка тестового набора...")
    test_dataset = TestDataset(csv_path, img_dir)
    print(f"   Найдено {len(test_dataset)} тестовых изображений")
    
    # Инициализация устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n2. Используемое устройство: {device}")
    
    # Загрузка моделей
    print("\n3. Загрузка моделей...")
    models_dict = load_models(device)
    if not models_dict:
        print("Ошибка: Не удалось загрузить модели. Выход.")
        return
    
    # Получение трансформаций
    transforms_dict = get_transforms()
    
    # Сохранение истинных меток и предсказаний
    true_labels = []
    model_predictions = {name: [] for name in models_dict.keys()}
    ensemble_predictions = []
    
    # Обработка каждого изображения
    print(f"\n4. Обработка {len(test_dataset)} тестовых изображений...")
    for i in range(len(test_dataset)):
        img_path, label = test_dataset[i]
        true_labels.append(label)
        
        # Пропуск отсутствующих файлов
        if not os.path.exists(img_path):
            print(f"Предупреждение: Изображение не найдено: {img_path}")
            # Добавление предсказаний по умолчанию для отсутствующих изображений
            for model_name in models_dict.keys():
                model_predictions[model_name].append(0)
            ensemble_predictions.append(0)
            continue
        
        # Предсказания от каждой модели для этого изображения
        image_predictions = []
        
        # Получение предсказания от каждой модели
        for model_name, model in models_dict.items():
            try:
                # Применение соответствующей трансформации
                transform = transforms_dict[model_name]
                
                # Открытие и трансформация изображения
                with Image.open(img_path).convert('L') as img:
                    input_tensor = transform(img).unsqueeze(0).to(device)
                
                # Получение предсказания
                with torch.no_grad():
                    if model_name == "inceptionv3":
                        output = model(input_tensor)
                        if isinstance(output, tuple):
                            output = output[0]
                    else:
                        output = model(input_tensor)
                    
                    _, pred = torch.max(output, 1)
                    pred = pred.item()
                    
                    # Сохранение предсказания
                    model_predictions[model_name].append(pred)
                    image_predictions.append(pred)
            except Exception as e:
                print(f"Ошибка обработки {img_path} с помощью {model_name}: {str(e)}")
                # Добавление предсказания по умолчанию в случае ошибки
                model_predictions[model_name].append(0)
                image_predictions.append(0)
        
        # Применение голосования большинства для этого изображения
        if image_predictions:
            ensemble_pred = majority_vote(image_predictions)
            ensemble_predictions.append(ensemble_pred)
        else:
            # Предсказание по умолчанию, если все модели не сработали
            ensemble_predictions.append(0)
        
        # Вывод прогресса
        if (i+1) % 100 == 0 or (i+1) == len(test_dataset):
            print(f"   Обработано {i+1}/{len(test_dataset)} изображений ({(i+1)/len(test_dataset)*100:.1f}%)")
    
    # Расчет метрик
    print("\n5. Расчет метрик производительности...")
    accuracy = accuracy_score(true_labels, ensemble_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, ensemble_predictions, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(true_labels, ensemble_predictions)
    
    # Построение матрицы ошибок
    plot_confusion_matrix(conf_matrix, class_names=['Отрицательный', 'Положительный'])
    
    # Вывод результатов ансамбля
    print("\n==== Производительность ансамблевой модели ====")
    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Матрица ошибок:\n{conf_matrix}")
    
    # Расчет производительности отдельных моделей
    print("\n==== Производительность отдельных моделей ====")
    results = []
    for name in models_dict.keys():
        model_acc = accuracy_score(true_labels, model_predictions[name])
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
            true_labels, model_predictions[name], average='binary', zero_division=0
        )
        results.append({
            'Model': name,
            'Accuracy': model_acc,
            'Precision': model_precision,
            'Recall': model_recall,
            'F1': model_f1
        })
        print(f"{name.upper()}: Accuracy={model_acc:.4f}, Precision={model_precision:.4f}, "
              f"Recall={model_recall:.4f}, F1={model_f1:.4f}")
    
    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    # Добавление результатов ансамбля
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'ensemble',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }])], ignore_index=True)
    results_df.to_csv('model_performance.csv', index=False)
    
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    evaluate_ensemble()
