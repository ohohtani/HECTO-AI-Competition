# ✅ DaViT 모델 사용 + Fine-tuning 적용 전체 코드 -> full_train.ver

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torchvision.transforms as transforms
import torch.nn.functional as F

import timm

# ===================== 설정 =====================
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 14,
    'LEARNING_RATE': 1e-4,
    'SEED': 42,
    'PATIENCE': 5,
    'MIXUP_ALPHA': 0.4
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

# ===================== MixUp 함수 =====================
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===================== 데이터셋 =====================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith('.jpg'):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith('.jpg'):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

# ===================== 전처리 =====================
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter()], p=0.3),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===================== 데이터 로딩 =====================
train_root = './filtered_train'
test_root = './test'

full_dataset = CustomImageDataset(train_root, transform=train_transform)
class_names = full_dataset.classes

train_loader = DataLoader(full_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

# ===================== DaViT 모델 정의 (Fine-tuning) =====================
class DaViTModel(nn.Module):
    def __init__(self, num_classes):
        super(DaViTModel, self).__init__()
        self.backbone = timm.create_model('davit_base', pretrained=True)
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        return self.backbone(x)

model = DaViTModel(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

# ===================== 학습(검증과정,early stopping 삭제) =====================
for epoch in range(CFG['EPOCHS']):
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        images, targets_a, targets_b, lam = mixup_data(images, labels, CFG['MIXUP_ALPHA'])
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")


# 마지막 epoch값 모델 저장
torch.save(model.state_dict(), 'final_model.pth')
print("✅ 최종 모델 저장 완료: final_model.pth")


# ===================== 추론 =====================
test_dataset = CustomImageDataset(test_root, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

model.load_state_dict(torch.load('final_model.pth', map_location=device))
model.to(device)
model.eval()

results = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        for prob in probs.cpu():
            result = {class_names[i]: prob[i].item() for i in range(len(class_names))}
            results.append(result)

pred = pd.DataFrame(results)
submission = pd.read_csv('sample_submission.csv', encoding='utf-8-sig')
class_columns = submission.columns[1:]
pred = pred[class_columns]
submission[class_columns] = pred.values
submission.to_csv('davit_submission_full_train_epoch14.csv', index=False, encoding='utf-8-sig')
print("✅ Submission saved to davit_submission.csv")
