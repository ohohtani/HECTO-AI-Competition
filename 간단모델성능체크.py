import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm

# ===================== 설정 =====================
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,
    'EPOCHS': 30,
    'LEARNING_RATE': 5e-5,
    'SEED': 42,
    'PATIENCE': 3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 시드 고정 =====================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

# ===================== 데이터셋 =====================
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'img_path']
        label = self.df.loc[idx, 'label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ===================== 전처리 =====================
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===================== 데이터 분할 =====================
train_df, val_df = train_test_split(filtered_train, test_size=0.2, stratify=filtered_train['label'], random_state=CFG['SEED'])

train_dataset = CustomDataset(train_df, transform=transform)
val_dataset = CustomDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# ===================== 모델 =====================
class DaViTModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('davit_base', pretrained=True)
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        return self.backbone(x)

model = DaViTModel(num_classes=filtered_train['label'].nunique()).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

# ===================== 학습 =====================
best_val_loss = float('inf')
best_epoch = 0
logloss_values = []

for epoch in range(CFG['EPOCHS']):
    model.train()
    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # === 검증 ===
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_logloss = log_loss(all_labels, all_probs, labels=list(range(filtered_train['label'].nunique())))
    logloss_values.append(val_logloss)
    print(f"Epoch {epoch+1}: LogLoss = {val_logloss:.4f}")

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"📦 Best model saved (Epoch {epoch+1}, LogLoss: {val_logloss:.4f})")
    elif epoch - best_epoch >= CFG['PATIENCE']:
        print(f"⛔ Early stopping at epoch {epoch+1}")
        break

# ===================== 그래프 시각화 =====================
plt.plot(range(1, len(logloss_values) + 1), logloss_values, marker='o')
plt.axvline(best_epoch + 1, color='red', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
plt.title('Validation LogLoss')
plt.xlabel('Epoch')
plt.ylabel('LogLoss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
