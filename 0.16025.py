import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torchvision.transforms as transforms
import torch.nn.functional as F

import timm

# ===================== 설정 =====================
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 25,
    'LEARNING_RATE': 2e-4,
    'SEED': 13,
    'MIXUP_ALPHA': 0.4,
    'CUTMIX_PROB': 0.5,
    'NUM_FOLDS': 5
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

# ===================== Transforms (RandomErasing 추가) =====================
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter()], p=0.3),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])
val_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_transform = val_transform  # 테스트는 검증과 동일

# ===================== 데이터 준비 =====================
full_dataset = CustomImageDataset('./filtered_train', transform=train_transform)
labels = np.array([lbl for _, lbl in full_dataset.samples])
skf    = StratifiedKFold(n_splits=CFG['NUM_FOLDS'], shuffle=True,
                         random_state=CFG['SEED'])

# ===================== K-Fold 학습 =====================
for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels), 1):
    print(f"\n====== Fold {fold} ======")
    seed_everything(CFG['SEED'] + fold)
    
    # Subset + transform 교체
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(
        CustomImageDataset('./filtered_train', transform=val_transform),
        val_idx
    )
    
    train_loader = DataLoader(train_ds, batch_size=CFG['BATCH_SIZE'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['BATCH_SIZE'], shuffle=False)
    
    # 모델 초기화
    model = timm.create_model('davit_base', pretrained=True,
                              num_classes=len(full_dataset.class_to_idx))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])
    criterion = nn.CrossEntropyLoss()
    
    best_loss, best_ep = np.inf, 0
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        total_loss, mix_cnt, cut_cnt = 0, 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"[F{fold} E{epoch}] Train"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            if random.random() < CFG['CUTMIX_PROB']:
                imgs, ya, yb, lam = cutmix_data(imgs, lbls, CFG['MIXUP_ALPHA'])
                cut_cnt += 1
            else:
                imgs, ya, yb, lam = mixup_data(imgs, lbls, CFG['MIXUP_ALPHA'])
                mix_cnt += 1
            out = model(imgs)
            loss = mixup_criterion(criterion, out, ya, yb, lam)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        # Validation
        model.eval()
        val_probs, val_lbls = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"[F{fold} E{epoch}] Val"):
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_probs.extend(F.softmax(out,1).cpu().numpy())
                val_lbls.extend(lbls.cpu().numpy())
        val_logloss = log_loss(val_lbls, val_probs,
                               labels=list(full_dataset.class_to_idx.values()))
        
        print(f"E{epoch}: TrainLoss={total_loss/len(train_loader):.4f}, "
              f"ValLogLoss={val_logloss:.4f}, MixUp={mix_cnt}, CutMix={cut_cnt}")
        
        if val_logloss < best_loss:
            best_loss, best_ep = val_logloss, epoch
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
    
    print(f"Fold{fold} ⇒ BestEpoch={best_ep}, LogLoss={best_loss:.4f}")

# ===================== 테스트 예측 앙상블 =====================
test_ds = CustomImageDataset('./test', transform=test_transform, is_test=True)
test_loader = DataLoader(test_ds, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# 클래스 순서 맞추기
sample_sub = pd.read_csv('sample_submission.csv', encoding='utf-8-sig')
columns = sample_sub.columns[1:]

# 각 Fold 예측값 누적
preds = np.zeros((len(test_ds), len(columns)), dtype=float)

for fold in range(1, CFG['NUM_FOLDS']+1):
    model = timm.create_model('davit_base', pretrained=False,
                              num_classes=len(full_dataset.class_to_idx))
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pth', map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        all_probs = []
        for imgs in tqdm(test_loader, desc=f"[Fold{fold}] Test"):
            imgs = imgs.to(device)
            out = model(imgs)
            all_probs.extend(F.softmax(out,1).cpu().numpy())
    preds += np.array(all_probs)

# 평균 확률로 최종 예측
preds /= CFG['NUM_FOLDS']
submission = sample_sub.copy()
submission[columns] = preds
submission.to_csv('davit_kfold_submission_erasing.csv', index=False, encoding='utf-8-sig')
print("✅ Finished K-Fold Ensemble Submission: davit_kfold_submission.csv")
