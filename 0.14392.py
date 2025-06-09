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
from sklearn.metrics import log_loss
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
skip_train = True          # ✅ 가중치가 이미 있을 때 True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device →', device)

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
#  MixUp / CutMix 함수 (학습용 — skip_train=False 때만 사용)
# =========================================================
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.size()
    cut_ratio = np.sqrt(1. - lam)
    cw, ch = int(W*cut_ratio), int(H*cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x[:, :, max(cy-ch//2,0):min(cy+ch//2,H),
           max(cx-cw//2,0):min(cx+cw//2,W)] = \
        x[idx, :, max(cy-ch//2,0):min(cy+ch//2,H),
                max(cx-cw//2,0):min(cx+cw//2,W)]
    lam = 1 - (cw * ch) / (W * H)
    return x, y, y[idx], lam

# =========================================================
#  Dataset
# =========================================================
class CustomImageDataset(Dataset):
    def __init__(self, root, tfm, is_test=False):
        self.t = tfm; self.test = is_test; self.samples=[]
        if is_test:
            for f in sorted(os.listdir(root)):
                if f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(root,f),))
        else:
            self.classes = sorted(os.listdir(root))
            cid = {c:i for i,c in enumerate(self.classes)}
            for c in self.classes:
                for f in os.listdir(os.path.join(root,c)):
                    if f.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(root,c,f), cid[c]))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        if self.test:
            p = self.samples[i][0]
            return self.t(Image.open(p).convert('RGB'))
        p, y = self.samples[i]
        return self.t(Image.open(p).convert('RGB')), y

# =========================================================
#  Transform
# =========================================================
mean,std=[0.485,0.456,0.406],[0.229,0.224,0.225]
train_t = transforms.Compose([
    HalfImageTransform(), transforms.Resize((CFG['IMG_SIZE']+32,)*2),
    transforms.RandomCrop(CFG['IMG_SIZE']), transforms.RandomHorizontalFlip(.5),
    transforms.RandomApply([transforms.ColorJitter(.3,.3,.3,.1)], p=.7),
    transforms.RandomAffine(15, translate=(.1,.1), scale=(.9,1.1)),
    transforms.RandomGrayscale(.1), transforms.RandAugment(num_ops=2, magnitude=5),
    transforms.ToTensor(), transforms.Normalize(mean,std),
    transforms.RandomErasing(p=.15, scale=(.02,.2), ratio=(.3,3.3))
])
val_t = transforms.Compose([transforms.Resize((CFG['IMG_SIZE'],)*2),
                            transforms.ToTensor(), transforms.Normalize(mean,std)])

# =========================================================
#  데이터 로드
# =========================================================
root = './train'
full_ds = CustomImageDataset(root, train_t)
labels  = [lbl for _, lbl in full_ds.samples]
classes = full_ds.classes

tr_idx, va_idx = train_test_split(range(len(full_ds)), test_size=.2,
                                  stratify=labels, random_state=CFG['SEED'])
train_loader = DataLoader(Subset(CustomImageDataset(root, train_t), tr_idx),
                          batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
val_loader   = DataLoader(Subset(CustomImageDataset(root, val_t),  va_idx),
                          batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# =========================================================
#  모델 & Freeze 설정
# =========================================================
model = timm.create_model('convnext_base', pretrained=True,
                          num_classes=len(classes), drop_path_rate=.1).to(device)

conv_names = [n for n,m in model.named_modules() if isinstance(m, nn.Conv2d)]
for n,m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        idx = conv_names.index(n)
        for p in m.parameters(): p.requires_grad = 5<=idx<=18
for n,m in model.named_modules():
    if isinstance(m,(nn.Linear, nn.LayerNorm)):
        for p in m.parameters(): p.requires_grad = True

# =========================================================
#  학습(선택) & 가중치 확보
# =========================================================
if not skip_train:
    criterion = nn.CrossEntropyLoss()
    opt  = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                       lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])
    scaler = GradScaler()

    class EMA:  # 간단 버전
        def __init__(self,m,d=0.9999):
            self.d=d; self.m=m; self.s={n:p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
        def update(self):
            for n,p in self.m.named_parameters():
                if p.requires_grad:
                    self.s[n].data = (1-self.d)*p.data + self.d*self.s[n].data
        def apply(self):
            self.b={}
            for n,p in self.m.named_parameters():
                if p.requires_grad:
                    self.b[n]=p.data.clone(); p.data=self.s[n]
        def restore(self):
            for n,p in self.m.named_parameters():
                if p.requires_grad and n in self.b: p.data=self.b[n]
    ema = EMA(model)

    best_log,no_imp=1e9,0
    for ep in range(1, CFG['EPOCHS']+1):
        # ---- Train ----
        model.train(); tl,correct,total=0,0,0
        for x,y in tqdm(train_loader, desc=f"[{ep}] Train"):
            x,y=x.to(device),y.to(device)
            if random.random()<CFG['CUTMIX_PROB']:
                x,ya,yb,lam=cutmix_data(x,y,CFG['MIXUP_ALPHA'])
                with autocast(): out=model(x); loss=lam*criterion(out,ya)+(1-lam)*criterion(out,yb)
            else:
                x,ya,yb,lam=mixup_data(x,y,CFG['MIXUP_ALPHA'])
                with autocast(): out=model(x); loss=lam*criterion(out,ya)+(1-lam)*criterion(out,yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            ema.update(); tl+=loss.item(); correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
        # ---- Val ----
        ema.apply(); model.eval()
        vl,corr,tot=0,0,0; probs,gts=[],[]
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc=f"[{ep}] Val"):
                x,y=x.to(device),y.to(device); out=model(x)
                vl+=criterion(out,y).item(); p=F.softmax(out,1)
                probs.extend(p.cpu().numpy()); gts.extend(y.cpu().numpy())
                corr+=(out.argmax(1)==y).sum().item(); tot+=y.size(0)
        ema.restore()
        logloss=log_loss(gts, probs, labels=list(range(len(classes))))
        print(f"Epoch {ep}  ValAcc {100*corr/tot:.2f}%  LogLoss {logloss:.4f}")

        if logloss<best_log:
            best_log,no_imp=logloss,0
            torch.save(model.state_dict(),'best_model.pth'); print("  ↳ saved best")
        else:
            no_imp+=1; print("  ↳ no improve", no_imp)
            if no_imp>=CFG['PATIENCE']: print("Early stop"); break

# =========================================================
#  추론 & CSV
# =========================================================
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device).eval()

test_root='./test'
test_loader=DataLoader(CustomImageDataset(test_root, val_t, is_test=True),
                       batch_size=CFG['BATCH_SIZE'], shuffle=False)
results=[]
with torch.no_grad():
    for imgs in tqdm(test_loader, desc='Inference'):
        imgs=imgs.to(device)
        probs=torch.softmax(model(imgs),1).cpu().numpy()
        results.append(probs)
pred=np.vstack(results)

# sample_submission 읽어 클래스 순서 맞추기
sub=pd.read_csv('sample_submission.csv', encoding='utf-8-sig')
sub[sub.columns[1:]]=pred[:, :len(sub.columns)-1]
csv_name=f"submission_convnext_{uuid.uuid4().hex[:6]}.csv"
sub.to_csv(csv_name, index=False, encoding='utf-8-sig')
print(f"✅ CSV 저장 완료 → {csv_name}")
