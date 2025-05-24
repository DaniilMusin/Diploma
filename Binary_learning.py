import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support

def main():
    # Параметры и пути
    CSV_PATH = r"D:/Diploma/labels_Denis.csv"
    IMG_DIR  = r"D:/Diploma/Plates/Processed_Grayscale_Images_Cropped"
    OUT_DIR  = r"D:/Diploma/Binary"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Загрузка данных
    data_list = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            if len(row) < 2:
                continue
            fname = row[0].strip()
            lbl_str = row[1].strip()
            try:
                lbl = int(lbl_str)
            except:
                continue
            if lbl not in [0, 1]:
                continue

            full_path = os.path.join(IMG_DIR, fname)
            if not os.path.isfile(full_path):
                print(f"Файл не найден: {full_path}")
                continue

            # Проверка целостности + исключение "пустых" изображений
            try:
                with Image.open(full_path) as test_img:
                    test_img.verify()
                with Image.open(full_path) as test_img2:
                    # Если вдруг файл не grayscale, можно конвертировать
                    if test_img2.mode != "L":
                        test_img2 = test_img2.convert("L")
                    img_array = np.array(test_img2)
                    # Если изображение слишком «пустое»
                    if img_array.std() < 5:
                        print(f"Пропускаем пустое изображение: {fname}")
                        continue
            except (UnidentifiedImageError, OSError) as e:
                print(f"Пропускаем поврежденный файл: {fname}, ошибка: {str(e)}")
                continue

            data_list.append((full_path, lbl))

    print(f"Всего валидных примеров: {len(data_list)}")
    
    if len(data_list) == 0:
        print("Ошибка: Не найдено ни одного валидного изображения для обучения.")
        return

    all_paths = np.array([d[0] for d in data_list])
    all_labels = np.array([d[1] for d in data_list], dtype=np.int64)

    # Анализ баланса классов
    class_counts = np.bincount(all_labels)
    print(f"Распределение классов: {class_counts}")

    # Вычисление весов для балансировки классов
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)
    class_weights = torch.FloatTensor(class_weights)
    print(f"Веса классов: {class_weights}")

    # Трансформации данных
    # Чтобы модели (resnet50, densenet121) корректно работали с pretrained-весами,
    # используем 3 канала (RGB). 
    # Либо добавим transforms.Grayscale(num_output_channels=3),
    # либо внутри Dataset сделаем convert("RGB").
    train_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),          # Превращаем 1 канал в 3
        transforms.Resize((256, 256)),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        # Стандартные mean/std для ImageNet (3 канала)
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Сет для классификации
    full_dataset = MyBinaryDataset(list(zip(all_paths, all_labels)), transform=None)

    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)

    EPOCHS = 20
    LR = 1e-4
    BATCH_SIZE = 32
    K = 3  # 3 фолда

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    # Будем обучать модели ResNet50 и DenseNet121
    results_dict = {"resnet50": [], "densenet121": []}

    fold_idx = 1
    for train_idx, val_idx in skf.split(all_paths, all_labels):
        print(f"\n===== Фолд {fold_idx}/{K} =====")
        train_subset = Subset(full_dataset, train_idx)
        val_subset   = Subset(full_dataset, val_idx)

        # Обёртка, которая применяет аугментации
        train_ds = TransformWrapper(train_subset, train_trans)
        val_ds   = TransformWrapper(val_subset,   val_trans)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        for arch_name in ["resnet50", "densenet121"]:
            print(f"Обучение {arch_name}, фолд={fold_idx}")
            model = get_model(arch_name)
            model.to(device)

            # Замораживаем все слои
            for param in model.parameters():
                param.requires_grad = False

            # Размораживаем только последний слой
            if arch_name == "resnet50":
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif arch_name == "densenet121":
                for param in model.classifier.parameters():
                    param.requires_grad = True

            # Критерий с весами классов
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            # Оптимизатор, обучающий только размороженные слои
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
            # Scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

            best_acc = 0.0
            best_f1 = 0.0

            log_file = os.path.join(OUT_DIR, f"{arch_name}_fold{fold_idx}_log.txt")
            with open(log_file, 'w', encoding='utf-8') as f_log:
                f_log.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_Precision,Val_Recall,Val_F1\n")

            # Этап 1: Начальная тренировка (замороженные слои)
            for epoch in range(1, EPOCHS + 1):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0
                
                for imgs, labels in train_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device, dtype=torch.long)

                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * imgs.size(0)
                    _, preds = torch.max(out, 1)
                    correct_train += (preds == labels).sum().item()
                    total_train += labels.size(0)

                train_loss = running_loss / total_train
                train_acc = correct_train / total_train

                # Валидация
                model.eval()
                correct_val = 0
                total_val = 0
                val_loss = 0.0
                all_preds = []
                all_true = []
                
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(device)
                        labels = labels.to(device, dtype=torch.long)
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * imgs.size(0)
                        
                        _, p = torch.max(outputs, 1)
                        correct_val += (p == labels).sum().item()
                        total_val += labels.size(0)
                        
                        all_preds.extend(p.cpu().numpy())
                        all_true.extend(labels.cpu().numpy())

                val_loss = val_loss / total_val
                val_acc = correct_val / total_val
                
                # Вычисляем precision, recall, f1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_true, all_preds, average='binary', zero_division=0
                )

                print(f"Эпоха [{epoch}/{EPOCHS}] train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                      f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
                
                with open(log_file, 'a', encoding='utf-8') as f_log:
                    f_log.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},"
                                f"{precision:.4f},{recall:.4f},{f1:.4f}\n")

                scheduler.step(val_acc)
                
                # Сохраняем лучшую модель по точности
                if val_acc > best_acc:
                    best_acc = val_acc
                    ckpt_path = os.path.join(OUT_DIR, f"best_acc_{arch_name}_fold{fold_idx}.pth")
                    torch.save(model.state_dict(), ckpt_path)

                # Лучшая модель по F1
                if f1 > best_f1:
                    best_f1 = f1
                    ckpt_path = os.path.join(OUT_DIR, f"best_f1_{arch_name}_fold{fold_idx}.pth")
                    torch.save(model.state_dict(), ckpt_path)

            # Этап 2: Fine-tuning (размораживаем все слои)
            for param in model.parameters():
                param.requires_grad = True

            # Снижаем LR
            optimizer = optim.Adam(model.parameters(), lr=LR / 10)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

            # Дополнительные 10 эпох
            for epoch in range(EPOCHS + 1, EPOCHS + 11):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for imgs, labels in train_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device, dtype=torch.long)

                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * imgs.size(0)
                    _, preds = torch.max(out, 1)
                    correct_train += (preds == labels).sum().item()
                    total_train += labels.size(0)

                train_loss = running_loss / total_train
                train_acc = correct_train / total_train

                model.eval()
                correct_val = 0
                total_val = 0
                val_loss = 0.0
                all_preds = []
                all_true = []

                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(device)
                        labels = labels.to(device, dtype=torch.long)
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * imgs.size(0)
                        
                        _, p = torch.max(outputs, 1)
                        correct_val += (p == labels).sum().item()
                        total_val += labels.size(0)
                        
                        all_preds.extend(p.cpu().numpy())
                        all_true.extend(labels.cpu().numpy())

                val_loss = val_loss / total_val
                val_acc = correct_val / total_val

                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_true, all_preds, average='binary', zero_division=0
                )

                print(f"Fine-tuning Эпоха [{epoch}/{EPOCHS + 10}] train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                      f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

                with open(log_file, 'a', encoding='utf-8') as f_log:
                    f_log.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},"
                                f"{precision:.4f},{recall:.4f},{f1:.4f}\n")

                scheduler.step(val_acc)

                if val_acc > best_acc:
                    best_acc = val_acc
                    ckpt_path = os.path.join(OUT_DIR, f"fine_tuned_acc_{arch_name}_fold{fold_idx}.pth")
                    torch.save(model.state_dict(), ckpt_path)

                if f1 > best_f1:
                    best_f1 = f1
                    ckpt_path = os.path.join(OUT_DIR, f"fine_tuned_f1_{arch_name}_fold{fold_idx}.pth")
                    torch.save(model.state_dict(), ckpt_path)

            results_dict[arch_name].append({"acc": best_acc, "f1": best_f1})
            print(f"{arch_name}, фолд={fold_idx}, best_acc={best_acc:.3f}, best_f1={best_f1:.3f}")

        fold_idx += 1

    print("\n=== Средние метрики по k-fold ===")
    for arch_name in results_dict:
        acc_values = [result["acc"] for result in results_dict[arch_name]]
        f1_values  = [result["f1"] for result in results_dict[arch_name]]
        
        mean_acc = np.mean(acc_values)
        std_acc  = np.std(acc_values)
        mean_f1  = np.mean(f1_values)
        std_f1   = np.std(f1_values)
        
        print(f"{arch_name}: mean_acc={mean_acc:.3f}±{std_acc:.3f}, mean_f1={mean_f1:.3f}±{std_f1:.3f} (K={K})")

        # Сохраняем итоги в файл
        results_file = os.path.join(OUT_DIR, f"{arch_name}_results.txt")
        with open(results_file, 'w', encoding='utf-8') as f_out:
            f_out.write(f"Архитектура: {arch_name}\n")
            f_out.write(f"K-fold: {K}\n")
            f_out.write(f"Средняя точность: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f_out.write(f"Средний F1-score: {mean_f1:.4f} ± {std_f1:.4f}\n\n")
            f_out.write("Результаты по фолдам:\n")
            for i, result in enumerate(results_dict[arch_name]):
                f_out.write(f"Фолд {i+1}: accuracy={result['acc']:.4f}, f1={result['f1']:.4f}\n")

def get_model(arch_name):
    """
    Загружаем модели ResNet50, DenseNet121 
    с предобученными весами на ImageNet (IMAGENET1K_V2).
    """
    import torch.nn as nn
    num_classes = 2

    if arch_name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Загружаем модель с весами
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        return net
    elif arch_name == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)  # Загружаем модель с весами
        in_f = net.classifier.in_features
        net.classifier = nn.Linear(in_f, num_classes)
        return net
    else:
        raise ValueError("Неизвестная модель: " + arch_name)

class MyBinaryDataset(Dataset):
    """
    Базовый датасет, в котором хранится список (путь к файлу, метка),
    и при __getitem__ выдаёт (img, label).
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, lbl = self.data_list[idx]
        # Загружаем изображение 
        # (Важно: Здесь уже как угодно, поскольку transform сам сделает Grayscale(3))
        img = Image.open(path).convert("L")  # прочитали grayscale
        if self.transform:
            img = self.transform(img)
        return img, lbl

class TransformWrapper(Dataset):
    """
    Обёртка, которая позволяет применить transform к существующему dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, lbl = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, lbl

if __name__ == "__main__":
    main()
