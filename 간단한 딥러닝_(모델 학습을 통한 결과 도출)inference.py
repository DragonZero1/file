import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import cv2
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 모델 로드 및 인퍼런스
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 랜덤으로 이미지를 수평 뒤집기
    transforms.RandomRotation(10),       # 랜덤으로 이미지 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색조 변환
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # 랜덤 크롭 및 사이즈 조정
    transforms.ToTensor()  # PIL 이미지를 텐서로 변환
])
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 인퍼런스 예시

def predict_image(image_path):
    global A                    #지역변수로 사용 안하고 전역 변수로 사용하고 싶을때, global A를 적어주면 된다. 이렇게 하면 함수 밖으로 빠져나가도 사용 할 수 있기 때문에! 
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    output = model(image)
    print(output)
    
    A, predicted = torch.max(output, 1)
    
    A=A.item()
    print(predicted)
    
    return predicted.item()

# 예시 이미지 경로
image_path = "bottle.jpg"
prediction = predict_image(image_path)
print(f"Predicted class: {prediction}_{A:.4f}") #f포맷터를 사용해서, 확률이 나오는 범위 즉, 자릿수를 지정 해주었다.

if prediction==1:
    
    img_src=cv2.imread('./data/1/20231206_135014.jpg')
    cv2.putText(img_src,f"Predicted class: {prediction}",(0,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,255),1 )
    cv2.putText(img_src,f"percentage:{A:.4f}",(0,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,255),1 )
    cv2.imshow('result',img_src)
    cv2.waitKey(0)
    
    