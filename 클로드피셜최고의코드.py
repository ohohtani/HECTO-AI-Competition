import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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
    'IMG_SIZE': 384,  # 더 큰 이미지 크기
    'BATCH_SIZE': 32,  # 배치 크기 줄임 (큰 이미지 때문에)
    'EPOCHS': 50,      # 에폭 증가
    'LEARNING_RATE': 1e-4,  # 학습률 낮춤
    'WEIGHT_DECAY': 1e-4,   # 가중치 감쇠 추가
    'SEED': 13,
    'MIXUP_ALPHA': 0.2,     # MixUp 강도 줄임
    'CUTMIX_PROB': 0.5,
    'VAL_RATIO': 0.15,      # 검증 비율
    'LABEL_SMOOTHING': 0.1,  # 라벨 스무딩 추가
    'TTA_CROPS': 5          # TTA용 크롭 수
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

# ===================== MixUp & CutMix =====================
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    B, C, W, H = x.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x[:, :, max(cx-cut_w//2,0):min(cx+cut_w//2,W),
         max(cy-cut_h//2,0):min(cy+cut_h//2,H)] = \
         x[idx, :, max(cx-cut_w//2,0):min(cx+cut_w//2,W),
                 max(cy-cut_h//2,0):min(cy+cut_h//2,H)]
    lam = 1 - (cut_w * cut_h) / (W * H)
    return x, y, y[idx], lam

def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)

# ===================== Dataset =====================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.transform = transform
        self.samples = []
        self.is_test = is_test
        if is_test:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith('.jpg'):
                    self.samples.append(os.path.join(root_dir, fname))
        else:
            classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
            for c in classes:
                for fname in os.listdir(os.path.join(root_dir, c)):
                    if fname.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(root_dir, c, fname),
                                              self.class_to_idx[c]))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        if self.is_test:
            path = self.samples[idx]
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return (self.transform(img), label) if self.transform else (img, label)

# ===================== 향상된 Transforms =====================
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# TTA용 변환
tta_transforms = [
    transforms.Compose([
        transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize((int(CFG['IMG_SIZE']*1.1), int(CFG['IMG_SIZE']*1.1))),
        transforms.CenterCrop(CFG['IMG_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize((int(CFG['IMG_SIZE']*1.2), int(CFG['IMG_SIZE']*1.2))),
        transforms.CenterCrop(CFG['IMG_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
]

# ===================== 모델 정의 (앙상블용) =====================
def create_model(model_name, num_classes):
    if model_name == 'convnext_large':
        model = timm.create_model('convnext_large', pretrained=True, num_classes=num_classes)
    elif model_name == 'swin_large':
        model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif model_name == 'davit_base':
        model = timm.create_model('davit_base', pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model('efficientnetv2_l', pretrained=True, num_classes=num_classes)
    return model

# ===================== 데이터 준비 =====================
full_dataset = CustomImageDataset('./filtered_train', transform=train_transform)
labels = np.array([lbl for _, lbl in full_dataset.samples])

# Train/Val 분할
train_idx, val_idx = train_test_split(
    np.arange(len(labels)), 
    test_size=CFG['VAL_RATIO'], 
    stratify=labels, 
    random_state=CFG['SEED']
)

# 데이터셋 생성
train_ds = torch.utils.data.Subset(full_dataset, train_idx)
val_ds = torch.utils.data.Subset(
    CustomImageDataset('./filtered_train', transform=val_transform), 
    val_idx
)

train_loader = DataLoader(train_ds, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)

print(f"학습 데이터: {len(train_ds)}, 검증 데이터: {len(val_ds)}")

# ===================== 다중 모델 학습 =====================
model_names = ['convnext_large', 'swin_large', 'davit_base']
trained_models = []

for model_name in model_names:
    print(f"\n====== {model_name} 학습 시작 ======")
    seed_everything(CFG['SEED'])
    
    # 모델 초기화
    model = create_model(model_name, len(full_dataset.class_to_idx))
    model.to(device)
    
    # 옵티마이저 및 스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG['LABEL_SMOOTHING'])
    
    best_loss = np.inf
    patience = 0
    max_patience = 10
    
    for epoch in range(1, CFG['EPOCHS']+1):
        # 학습
        model.train()
        total_loss = 0
        mix_cnt, cut_cnt = 0, 0
        
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            # 데이터 증강 적용
            if random.random() < CFG['CUTMIX_PROB']:
                imgs, ya, yb, lam = cutmix_data(imgs, lbls, CFG['MIXUP_ALPHA'])
                cut_cnt += 1
            else:
                imgs, ya, yb, lam = mixup_data(imgs, lbls, CFG['MIXUP_ALPHA'])
                mix_cnt += 1
            
            out = model(imgs)
            loss = mixup_criterion(criterion, out, ya, yb, lam)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # 검증
        model.eval()
        val_probs, val_lbls = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"[{model_name}] Validation"):
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_probs.extend(F.softmax(out, 1).cpu().numpy())
                val_lbls.extend(lbls.cpu().numpy())
        
        val_logloss = log_loss(val_lbls, val_probs, labels=list(full_dataset.class_to_idx.values()))
        
        print(f"Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f}, "
              f"Val LogLoss={val_logloss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Early Stopping
        if val_logloss < best_loss:
            best_loss = val_logloss
            patience = 0
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            print(f"✅ Best model saved! LogLoss: {best_loss:.4f}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 최고 모델 로드
    model.load_state_dict(torch.load(f'best_{model_name}.pth'))
    trained_models.append((model_name, model))
    print(f"{model_name} 최종 성능: {best_loss:.4f}")

# ===================== TTA를 활용한 테스트 예측 =====================
sample_sub = pd.read_csv('sample_submission.csv', encoding='utf-8-sig')
columns = sample_sub.columns[1:]

# 각 모델별 예측값을 저장할 배열
all_predictions = []

for model_name, model in trained_models:
    print(f"\n====== {model_name} TTA 예측 ======")
    model.eval()
    
    model_preds = np.zeros((len(sample_sub), len(columns)))
    
    # TTA 적용
    for tta_idx, tta_transform in enumerate(tta_transforms):
        print(f"TTA {tta_idx+1}/{len(tta_transforms)}")
        
        test_ds = CustomImageDataset('./test', transform=tta_transform, is_test=True)
        test_loader = DataLoader(test_ds, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)
        
        tta_preds = []
        with torch.no_grad():
            for imgs in tqdm(test_loader):
                imgs = imgs.to(device)
                out = model(imgs)
                tta_preds.extend(F.softmax(out, 1).cpu().numpy())
        
        model_preds += np.array(tta_preds)
    
    # TTA 평균
    model_preds /= len(tta_transforms)
    all_predictions.append(model_preds)

# ===================== 앙상블 예측 =====================
# 가중 평균 (성능 기반 가중치)
weights = [0.4, 0.35, 0.25]  # ConvNext, Swin, DaViT 순서
final_preds = np.zeros_like(all_predictions[0])

for i, (pred, weight) in enumerate(zip(all_predictions, weights)):
    final_preds += pred * weight

# ===================== 제출 파일 생성 =====================
submission = sample_sub.copy()
submission[columns] = final_preds
submission.to_csv('ensemble_tta_submission.csv', index=False, encoding='utf-8-sig')

print("✅ 앙상블 + TTA 제출 파일 생성 완료: ensemble_tta_submission.csv")
print(f"예측 확률 범위: {final_preds.min():.6f} ~ {final_preds.max():.6f}")
print(f"각 행의 확률 합계 평균: {final_preds.sum(axis=1).mean():.6f}")
