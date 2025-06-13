# =========================================================
#  추론 전용 코드 
# =========================================================
import os, uuid
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as transforms

# =========================================================
#  설정
# =========================================================
CFG = {
    'IMG_SIZE': 416,
    'BATCH_SIZE': 32,
    'MODEL_PATH': 'best_model_416.pth',  # 가중치 파일 경로
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device →', device)

# =========================================================
#  Dataset (테스트용)
# =========================================================
class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.samples = []
        for f in sorted(os.listdir(root)):
            if f.lower().endswith('.jpg'):
                self.samples.append(os.path.join(root, f))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

# =========================================================
#  Transform (추론용)
# =========================================================
test_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================================================
#  모델 로드
# =========================================================
# 클래스 수 확인 (sample_submission.csv에서)
sub = pd.read_csv('sample_submission.csv', encoding='utf-8-sig')
num_classes = len(sub.columns) - 1  # ID 컬럼 제외

# 모델 생성 및 가중치 로드
model = timm.create_model('convnext_base', pretrained=False,
                          num_classes=num_classes, drop_path_rate=0.1)
model.load_state_dict(torch.load(CFG['MODEL_PATH'], map_location=device))
model.to(device).eval()

# =========================================================
#  추론 실행
# =========================================================
test_dataset = TestDataset('./test', test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], 
                        shuffle=False, num_workers=0)

predictions = []
with torch.no_grad():
    for imgs in tqdm(test_loader, desc='추론 중...'):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions.append(probs)

# 결과 합치기
pred_array = np.vstack(predictions)

# =========================================================
#  CSV 생성
# =========================================================
sub[sub.columns[1:]] = pred_array
csv_name = f"submission_convnext_{uuid.uuid4().hex[:6]}.csv"
sub.to_csv(csv_name, index=False, encoding='utf-8-sig')
print(f"✅ 추론 완료! CSV 저장 → {csv_name}")
print(f"📊 총 {len(test_dataset)}개 이미지 처리됨")
