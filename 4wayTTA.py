# =========================================================
#  라이브러리
# =========================================================
import os, random, uuid, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import timm

# =========================================================
#  설정
# =========================================================
CFG = {
    'IMG_SIZE'    : 384,
    'BATCH_SIZE'  : 32,
    'SEED'        : 42,

    'SKIP_TRAIN'  : True,
    'WEIGHT_PATH' : 'best_model.pth',        # conv#5~18 finetune 가중치
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device →', device)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
seed_everything(CFG['SEED'])

# =========================================================
#  Dataset
# =========================================================
class CustomImageDataset(Dataset):
    def __init__(self, root, is_test=False):
        self.is_test, self.samples = is_test, []
        if is_test:
            for f in sorted(os.listdir(root)):
                if f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(root, f),))
        else:
            self.classes = sorted(os.listdir(root))
            cid = {c:i for i,c in enumerate(self.classes)}
            for c in self.classes:
                for f in os.listdir(os.path.join(root,c)):
                    if f.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(root,c,f), cid[c]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            p = self.samples[idx][0]
            return Image.open(p).convert('RGB'), os.path.basename(p)
        p,y = self.samples[idx]
        return Image.open(p).convert('RGB'), y

# --------------------------- ▲ 수정 : collate ----------------------------
def collate_pil(batch):
    """batch →  (list[PIL], list[id])   (id는 파일명 또는 라벨)"""
    imgs, ids = zip(*batch)
    return list(imgs), list(ids)
# ------------------------------------------------------------------------

# =========================================================
#  Transform & 4-way TTA
# =========================================================
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
def rot_tfm(angle):
    return T.Compose([
        T.Resize((CFG['IMG_SIZE'],)*2),
        T.Lambda(lambda img: TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)),
        T.ToTensor(), T.Normalize(MEAN,STD)
    ])

TTA_LIST = [
    T.Compose([T.Resize((CFG['IMG_SIZE'],)*2),
               T.ToTensor(), T.Normalize(MEAN,STD)]),                     # 원본
    T.Compose([T.Resize((CFG['IMG_SIZE'],)*2),
               T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize(MEAN,STD)]),                    # 좌우
    rot_tfm(+5),                                                        # +5°
    rot_tfm(-5)                                                         # –5°
]

# =========================================================
#  DataLoader (test only)
# =========================================================
test_ds = CustomImageDataset('./test', is_test=True)
test_loader = DataLoader(test_ds,
                         batch_size=CFG['BATCH_SIZE'],
                         shuffle=False,
                         num_workers=0,
                         collate_fn=collate_pil)          # ▲ 수정

# ---------------------------------------------------------
#  클래스 정보 (train 폴더 스캔)
# ---------------------------------------------------------
train_root = './train'
classes    = sorted(os.listdir(train_root)); N = len(classes)
print("#Classes =", N)

# =========================================================
#  모델
# =========================================================
model = timm.create_model('convnext_base',
                          pretrained=False,
                          num_classes=N, drop_path_rate=.1).to(device)
model.load_state_dict(torch.load(CFG['WEIGHT_PATH'], map_location=device))
model.eval()
print("✅ weights loaded")

# =========================================================
#  4-way TTA inference
# =========================================================
all_probs = []

with torch.no_grad():
    for pil_imgs, fnames in tqdm(test_loader, desc="Inference"):
        B = len(pil_imgs)
        prob_sum = torch.zeros(B, N, device=device)

        for tfm in TTA_LIST:
            batch = torch.stack([tfm(img) for img in pil_imgs]).to(device)
            prob_sum += F.softmax(model(batch), 1)

        all_probs.append((prob_sum/len(TTA_LIST)).cpu())

probs = torch.cat(all_probs).numpy()

# =========================================================
#  CSV
# =========================================================
sub = pd.read_csv('sample_submission.csv', encoding='utf-8-sig')
sub[sub.columns[1:]] = probs
csv_name = f'sub_4wayTTA_{uuid.uuid4().hex[:6]}.csv'
sub.to_csv(csv_name, index=False, encoding='utf-8-sig')
print("✅ CSV 저장 완료 →", csv_name)
