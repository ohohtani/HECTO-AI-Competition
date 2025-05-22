# ✅ DaViT 모델 + Fine-tuning + MixUp + CutMix + Label Smoothing + TTA 적용 전체 코드

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
    'EPOCHS': 8,
    'LEARNING_RATE': 1e-4,
    'SEED': 42,
    'PATIENCE': 2,
    'MIXUP_ALPHA': 0.4,
    'CUTMIX_ALPHA': 1.0
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

# ===================== MixUp / CutMix 함수 =====================
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

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
            # 테스트 모드에서는 클래스 폴더 없이 파일만 존재한다고 가정
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')

            if self.transform is None:
                raise ValueError("transform must be provided in test mode to convert PIL image to tensor.")

            image = self.transform(image)
            return image

        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            else:
                raise ValueError("transform must be provided in training/validation mode as well.")

            return image, label

# ===================== 전처리 =====================
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter()], p=0.3),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 데이터 로딩 =====================
train_root = '/content/filtered_train'
test_root = '/content/test'

full_dataset = CustomImageDataset(train_root)
targets = [label for _, label in full_dataset.samples]
class_names = full_dataset.classes

train_idx, val_idx = train_test_split(range(len(targets)), test_size=0.2, stratify=targets, random_state=CFG['SEED'])
train_dataset = Subset(CustomImageDataset(train_root, transform=train_transform), train_idx)
val_dataset = Subset(CustomImageDataset(train_root, transform=val_transform), val_idx)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# ===================== DaViT 모델 정의 =====================
class DaViTModel(nn.Module):
    def __init__(self, num_classes):
        super(DaViTModel, self).__init__()
        self.backbone = timm.create_model('davit_base', pretrained=True)
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        return self.backbone(x)

model = DaViTModel(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

# ===================== TTA 유효성 검사 함수 =====================
def validate_tta_transforms(tta_transforms, img_size):
    from torchvision.transforms import ToPILImage
    dummy = torch.rand(3, img_size, img_size)
    for i, t in enumerate(tta_transforms):
        try:
            pil = ToPILImage()(dummy)
            _ = t(pil)
        except Exception as e:
            print(f"\n❌ TTA Transform #{i+1} failed: {e}")
            raise RuntimeError("🚫 TTA transform 충돌로 인해 실행 중단됨.")
    print("\n✅ All TTA transforms validated successfully.\n")

# ===================== 학습 =====================
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(CFG['EPOCHS']):
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        r = np.random.rand()
        if r < 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, labels, CFG['MIXUP_ALPHA'])
        else:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, CFG['CUTMIX_ALPHA'])

        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%")

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"\n📦 Saved best model (logloss: {val_logloss:.4f}) at epoch {epoch+1}")
    elif epoch - best_epoch >= CFG['PATIENCE']:
        print(f"\n⛔ EarlyStopping at epoch {epoch+1}")
        break

# ===================== TTA 추론 =====================
test_dataset = CustomImageDataset(test_root, transform=None, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# 여러 TTA 변형 적용
tta_transforms = [
    val_transform,
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), *val_transform.transforms]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), *val_transform.transforms]),
    transforms.Compose([transforms.RandomRotation(15), *val_transform.transforms])
]

results = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        tta_probs = []

        for tta in tta_transforms:
            augmented_imgs = torch.stack([tta(transforms.ToPILImage()(img.cpu())) for img in images]).to(device)
            outputs = model(augmented_imgs)
            probs = F.softmax(outputs, dim=1)
            tta_probs.append(probs.cpu().numpy())

        avg_probs = np.mean(tta_probs, axis=0)
        for prob in avg_probs:
            result = {class_names[i]: prob[i].item() for i in range(len(class_names))}
            results.append(result)

pred = pd.DataFrame(results)
submission = pd.read_csv('/content/sample_submission.csv', encoding='utf-8-sig')
class_columns = submission.columns[1:]
pred = pred[class_columns]
submission[class_columns] = pred.values
submission.to_csv('/content/davit_submission.csv', index=False, encoding='utf-8-sig')
print("✅ Submission saved to davit_submission.csv")
