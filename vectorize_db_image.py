import os
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import json

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = nn.Sequential(*list(model.children())[:-1])

image_dir = "./sample_image"
image_dict = dict()

# 디렉토리 내의 모든 파일에 대해 반복
for file_path in sorted(os.listdir(image_dir)):
    # 파일의 확장자가 이미지인지 확인
    # print(file_path)
    if file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
        input_image = Image.open(os.path.join(image_dir, file_path))
        input_image = input_image.convert("RGB")
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가
        model.eval()
        with torch.no_grad():
            feature_vector = model(input_batch)
            image_dict[file_path] = feature_vector.squeeze().cpu().numpy().tolist()

# 저장할 파일 경로
json_path = 'image_dict.json'

# 딕셔너리를 JSON 형식으로 변환하여 파일에 저장
with open(json_path, 'w') as f:
    json.dump(image_dict, f)

# 저장된 파일에서 딕셔너리 읽어오기
with open(json_path, 'r') as f:
    loaded_dict = json.load(f)

# 로드된 딕셔너리 확인
# print(loaded_dict)
