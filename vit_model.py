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
import torch.cuda.amp as amp


# ================ Vision Transformer Implementation ================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim должен быть кратен num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=2,
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_recursive)
    
    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        
        return x


def create_vit_base(num_classes=2):
    return ViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )


# ================ Dataset Classes ================
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


def get_model(arch_name):
    if arch_name == "vit_base":
        return create_vit_base(num_classes=2)
    else:
        raise ValueError(f"Неизвестная модель: {arch_name}")


def main():
    CSV_PATH = r"D:\Diploma\labels_Denis.csv"
    IMG_DIR = r"D:\Diploma\Plates"
    OUT_DIR = "checkpoints_vit"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Загрузка и фильтрация данных
    data_list = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            if len(row) < 2: continue
            fname, lbl_str = row[0].strip(), row[1].strip()
            try: lbl = int(lbl_str)
            except: continue
            if lbl not in [0,1]: continue
            full_path = os.path.join(IMG_DIR, fname)
            if not os.path.isfile(full_path): continue
            try:
                with Image.open(full_path) as test_img:
                    test_img.verify()
            except (UnidentifiedImageError, OSError):
                print(f"Пропускаем битый файл: {fname}")
                continue
            data_list.append((full_path, lbl))

    print(f"Всего годных примеров: {len(data_list)}")

    all_paths = np.array([d[0] for d in data_list])
    all_labels = np.array([d[1] for d in data_list], dtype=np.int64)

    # Оптимизированные аугментации для ViT
    train_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomAffine(degrees=30, shear=15),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = MyBinaryDataset(list(zip(all_paths, all_labels)), transform=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    # Параметры для RTX 3080 (10GB VRAM)
    EPOCHS = 50
    BATCH_SIZE = 64  # Максимально возможный размер батча
    LR = 3e-5
    K = 5

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    fold_idx = 1
    for train_idx, val_idx in skf.split(all_paths, all_labels):
        print(f"\n===== Fold {fold_idx}/{K} =====")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_ds = TransformWrapper(train_subset, train_trans)
        val_ds = TransformWrapper(val_subset, val_trans)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

        # Инициализация модели и оптимизатора - БЕЗ torch.compile()
        model = get_model("vit_base")
        model = model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
        scaler = amp.GradScaler()
        
        best_acc = 0.0
        for epoch in range(1, EPOCHS+1):
            # Training
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_loss = total_loss / total

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with amp.autocast():
                        outputs = model(images)
                    
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            print(f"Epoch [{epoch}/{EPOCHS}] | Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = os.path.join(OUT_DIR, f"vit_base_fold{fold_idx}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, ckpt_path)
                print(f"Улучшение! Сохранен чекпоинт: {ckpt_path}")

        print(f"Лучшая точность для fold {fold_idx}: {best_acc:.4f}")
        fold_idx += 1


if __name__ == "__main__":
    main()
