# =========================================================
#  라이브러리
# =========================================================
import os, random, uuid, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch, torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import timm, torchvision.transforms as transforms

# =========================================================
#  설정
# =========================================================
CFG = {
    'IMG_SIZE'    : 384,
    'BATCH_SIZE'  : 32,
    'EPOCHS'      : 50,
    'LEARNING_RATE': 1e-4,
    'WEIGHT_DECAY': 2e-4,
    'SEED'        : 42,
    'PATIENCE'    : 8,
    'MIXUP_ALPHA' : 0.4,
    'CUTMIX_PROB' : 0.4,
    'HALF_PROB'   : 0.5,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device →', device)

# =========================================================
#  EMA
# =========================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.model  = model
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n].data = (1. - self.decay) * p.data + self.decay * self.shadow[n].data

    def apply_shadow(self):
        self.backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n].data

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]
        self.backup = {}

# =========================================================
#  Seed 고정
# =========================================================
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
seed_everything(CFG['SEED'])

# =========================================================
#  Half-crop 변환
# =========================================================
class HalfImageTransform:
    def __call__(self, img):
        if random.random() < CFG['HALF_PROB']:
            w, h = img.size
            if random.random() < .5:
                img = img.crop((0, 0, w // 2, h))
            else:
                img = img.crop((w // 2, 0, w, h))
        return img

# =========================================================
#  MixUp / CutMix
# =========================================================
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.size()
    cut_ratio = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x[:, :, max(cy-cut_h//2,0):min(cy+cut_h//2,H),
           max(cx-cut_w//2,0):min(cx+cut_w//2,W)] = \
        x[idx, :, max(cy-cut_h//2,0):min(cy+cut_h//2,H),
                max(cx-cut_w//2,0):min(cx+cut_w//2,W)]
    lam = 1 - (cut_w * cut_h) / (W * H)
    return x, y, y[idx], lam

# =========================================================
#  데이터셋
# =========================================================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform, is_test=False):
        self.t = transform; self.is_test = is_test; self.samples=[]
        if is_test:
            for f in sorted(os.listdir(root_dir)):
                if f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(root_dir, f),))
        else:
            self.classes = sorted(os.listdir(root_dir))
            cid = {c: i for i, c in enumerate(self.classes)}
            for c in self.classes:
                for f in os.listdir(os.path.join(root_dir, c)):
                    if f.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(root_dir, c, f), cid[c]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            p = self.samples[idx][0]
            return self.t(Image.open(p).convert('RGB'))
        p, y = self.samples[idx]
        return self.t(Image.open(p).convert('RGB')), y

# =========================================================
#  Transform
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_t = transforms.Compose([
    HalfImageTransform(),
    transforms.Resize((CFG['IMG_SIZE']+32,)*2),
    transforms.RandomCrop(CFG['IMG_SIZE']),
    transforms.RandomHorizontalFlip(.5),
    transforms.RandomApply([transforms.ColorJitter(.3,.3,.3,.1)], p=.7),
    transforms.RandomAffine(15, translate=(.1,.1), scale=(.9,1.1)),
    transforms.RandomGrayscale(.1),
    transforms.RandAugment(num_ops=2, magnitude=5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=.15, scale=(.02,.2), ratio=(.3,3.3))
])
val_t = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'],)*2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# =========================================================
#  데이터 로드
# =========================================================
root = './train'
full_ds = CustomImageDataset(root, transform=train_t)
labels  = [lbl for _, lbl in full_ds.samples]
classes = full_ds.classes

tr_idx, va_idx = train_test_split(range(len(full_ds)), test_size=.2,
                                  stratify=labels, random_state=CFG['SEED'])

train_loader = DataLoader(
    Subset(CustomImageDataset(root, train_t), tr_idx),
    batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0
)
val_loader = DataLoader(
    Subset(CustomImageDataset(root, val_t), va_idx),
    batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0
)

# =========================================================
#  모델 + Freeze/Un-freeze 설정
# =========================================================
model = timm.create_model('convnext_base', pretrained=True,
                          num_classes=len(classes), drop_path_rate=.1).to(device)

conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
assert len(conv_names) == 40

for name, mod in model.named_modules():
    if isinstance(mod, nn.Conv2d):
        idx = conv_names.index(name)
        trainable = 5 <= idx <= 18
        for p in mod.parameters():
            p.requires_grad = trainable

for n, m in model.named_modules():          # head & LayerNorm 학습
    if isinstance(m, (nn.Linear, nn.LayerNorm)):
        for p in m.parameters():
            p.requires_grad = True

# =========================================================
#  Optimizer / AMP / EMA
# =========================================================
opt     = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])
scaler  = GradScaler()
ema     = EMA(model, decay=0.9999)

criterion = nn.CrossEntropyLoss()

# =========================================================
#  학습 루프
# =========================================================
tr_loss, va_loss, tr_acc, va_acc = [], [], [], []
best_log, no_imp = float('inf'), 0

for epoch in range(1, CFG['EPOCHS'] + 1):
    # ---------- Train ----------
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in tqdm(train_loader, desc=f"[{epoch}] Train"):
        x, y = x.to(device), y.to(device)

        if random.random() < CFG['CUTMIX_PROB']:
            x, ya, yb, lam = cutmix_data(x, y, CFG['MIXUP_ALPHA'])
            with autocast(): out = model(x); loss = lam*criterion(out, ya)+(1-lam)*criterion(out, yb)
        else:
            x, ya, yb, lam = mixup_data(x, y, CFG['MIXUP_ALPHA'])
            with autocast(): out = model(x); loss = lam*criterion(out, ya)+(1-lam)*criterion(out, yb)

        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad()
        ema.update()

        loss_sum += loss.item()
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)

    tr_loss.append(loss_sum / len(train_loader))
    tr_acc.append(100 * correct / total)

    # ---------- Val ----------
    model.eval(); ema.apply_shadow()
    v_loss_sum, correct, total = 0, 0, 0; probs, gts = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"[{epoch}] Val"):
            x, y = x.to(device), y.to(device)
            out   = model(x)
            v_loss_sum += criterion(out, y).item()
            p = F.softmax(out, 1)
            probs.extend(p.cpu().numpy()); gts.extend(y.cpu().numpy())
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
    ema.restore()

    va_loss.append(v_loss_sum / len(val_loader))
    va_acc.append(100 * correct / total)
    logloss = log_loss(gts, probs, labels=list(range(len(classes))))

    print(f"Epoch {epoch:02d} | "
          f"Train {tr_loss[-1]:.4f}/{tr_acc[-1]:.2f}% || "
          f"Val {va_loss[-1]:.4f}/{va_acc[-1]:.2f}%")

    if logloss < best_log:
        best_log, no_imp = logloss, 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("  ↳ best saved (logloss ↓)")
    else:
        no_imp += 1
        print(f"  ↳ no improve {no_imp}/{CFG['PATIENCE']}")
        if no_imp >= CFG['PATIENCE']:
            print("Early stopping"); break

# =========================================================
#  Plotly 그래프
# =========================================================
fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
fig.add_trace(go.Scatter(y=tr_loss, name='Train Loss'), row=1, col=1)
fig.add_trace(go.Scatter(y=va_loss, name='Val Loss'),   row=1, col=1)
fig.add_trace(go.Scatter(y=tr_acc, name='Train Acc'),   row=1, col=2)
fig.add_trace(go.Scatter(y=va_acc, name='Val Acc'),     row=1, col=2)
fig.update_xaxes(title='Epoch'); fig.update_yaxes(title='Value')
fig.update_layout(title='Training / Validation Curves', width=950)
fig.show()

print("✅ 완료 – 최적 가중치 'best_model_train.pth' 저장")
