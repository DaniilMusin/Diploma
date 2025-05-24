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



def main():

    """

    K-fold (k=5) на задачу бинарной классификации (0/1).

    Обучаем 4 архитектуры: ResNet101, EffNetB7, ConvNeXt Large, DenseNet121.

    Сохраняем лучший чекпоинт на каждом фолде.

    """



    CSV_PATH = r"D:\Diploma\labels_Denis.csv"

    IMG_DIR  = r"D:\Diploma\Plates"

    OUT_DIR  = "checkpoints_kfold"

    os.makedirs(OUT_DIR, exist_ok=True)



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

            if lbl not in [0,1]:

                continue



            full_path = os.path.join(IMG_DIR, fname)

            if not os.path.isfile(full_path):

                continue



            try:

                with Image.open(full_path) as test_img:

                    test_img.verify()

            except (UnidentifiedImageError, OSError):

                print(f"Пропускаем битый файл: {fname}")

                continue



            data_list.append((full_path, lbl))



    print(f"Всего годных примеров: {len(data_list)}")

    import numpy as

    # Важно: np уже импортирован

    all_paths  = np.array([d[0] for d in data_list])

    all_labels = np.array([d[1] for d in data_list], dtype=np.int64)



    MODEL_LIST = ["resnet101","efficientnet_b7","convnext_large","densenet121"]



    import torchvision.transforms as T

    train_trans = T.Compose([

        T.Resize((256,256)),

        T.RandomRotation(30),

        T.RandomPerspective(distortion_scale=0.4, p=0.5),

        T.RandomHorizontalFlip(0.5),

        T.RandomVerticalFlip(0.2),

        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),

        T.RandomResizedCrop(224, scale=(0.5,1.0)),

        T.ToTensor(),

        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ])

    val_trans = T.Compose([

        T.Resize((224,224)),

        T.ToTensor(),

        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ])



    full_dataset = MyBinaryDataset(list(zip(all_paths, all_labels)), transform=None)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)



    EPOCHS=30

    LR=1e-4

    BATCH_SIZE=32

    K=5



    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)



    results_dict = {m:[] for m in MODEL_LIST}



    fold_idx=1

    for train_idx, val_idx in skf.split(all_paths, all_labels):

        print(f"\n===== Fold {fold_idx}/{K} =====")

        train_subset = Subset(full_dataset, train_idx)

        val_subset   = Subset(full_dataset, val_idx)



        train_ds = TransformWrapper(train_subset, train_trans)

        val_ds   = TransformWrapper(val_subset,   val_trans)



        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



        for arch_name in MODEL_LIST:

            print(f"Обучаем {arch_name}, fold={fold_idx}")

            model = get_model(arch_name)

            model.to(device)



            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(model.parameters(), lr=LR)



            best_acc = 0.0

            for epoch in range(1,EPOCHS+1):

                # train

                model.train()

                running_loss=0.0

                correct_train=0

                total_train=0

                for imgs, labels in train_loader:

                    imgs = imgs.to(device)

                    labels = labels.to(device, dtype=torch.long)



                    optimizer.zero_grad()

                    out = model(imgs)

                    loss = criterion(out, labels)

                    loss.backward()

                    optimizer.step()



                    running_loss += loss.item()*imgs.size(0)

                    _, preds = torch.max(out,1)

                    correct_train += (preds==labels).sum().item()

                    total_train   += labels.size(0)

                train_acc = correct_train/total_train



                # val

                model.eval()

                correct_val=0

                total_val=0

                with torch.no_grad():

                    for imgs, labels in val_loader:

                        imgs = imgs.to(device)

                        labels= labels.to(device, dtype=torch.long)

                        outputs = model(imgs)

                        _, p = torch.max(outputs, 1)

                        correct_val += (p==labels).sum().item()

                        total_val += labels.size(0)

                val_acc = correct_val/total_val



                print(f"  Epoch [{epoch}/{EPOCHS}] train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

                if val_acc>best_acc:

                    best_acc= val_acc

                    ckpt_path = os.path.join(OUT_DIR, f"best_{arch_name}_fold{fold_idx}.pth")

                    torch.save(model.state_dict(), ckpt_path)



            results_dict[arch_name].append(best_acc)

            print(f"{arch_name}, fold={fold_idx}, best_acc={best_acc:.3f}")



        fold_idx+=1



    # Итог

    import numpy as np

    print("\n=== Средняя точность по k-fold ===")

    for arch_name in MODEL_LIST:

        arr = results_dict[arch_name]

        mean_acc = np.mean(arr)

        std_acc  = np.std(arr)

        print(f"{arch_name}: mean={mean_acc:.3f}, std={std_acc:.3f} (на {K} фолдах)")





def get_model(arch_name):

    import torchvision.models as M

    import torch.nn as nn

    num_classes=2



    if arch_name=="resnet101":

        net = M.resnet101(weights=None)

        in_f= net.fc.in_features

        net.fc= nn.Linear(in_f,num_classes)

        return net

    elif arch_name=="efficientnet_b7":

        net = M.efficientnet_b7(weights=None)

        in_f= net.classifier[1].in_features

        net.classifier[1] = nn.Linear(in_f,num_classes)

        return net

    elif arch_name=="convnext_large":

        net = M.convnext_large(weights=None)

        in_f= net.classifier[2].in_features

        net.classifier[2]=nn.Linear(in_f,num_classes)

        return net

    elif arch_name=="densenet121":

        net = M.densenet121(weights=None)

        in_f= net.classifier.in_features

        net.classifier=nn.Linear(in_f,num_classes)

        return net

    else:

        raise ValueError("Неизвестная модель:"+arch_name)





class MyBinaryDataset(Dataset):

    def __init__(self, data_list, transform=None):

        self.data_list = data_list

        self.transform = transform



    def __len__(self):

        return len(self.data_list)



    def __getitem__(self, idx):

        path, lbl = self.data_list[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:

            img = self.transform(img)

        return img, lbl



class TransformWrapper(Dataset):

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





if __name__=="__main__":

    main()





