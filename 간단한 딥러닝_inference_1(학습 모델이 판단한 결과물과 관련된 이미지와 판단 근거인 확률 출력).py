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
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    output = model(image) #2차원 텐서로서, 첫번재 열에는, 양성 클래서, 두번째 열에는 음성 클래스가 나오게 된다.
    print(output)   #tensor([[ 0.0784, -0.0299]] , grad_fn=<AddmmBackward0>)
                    #이게 지금 출력값이 2차원 배열의 형태로 나오게 되는데, 
                    #각 숫자는 모델이 예측한 클래스에 대한 출력값을 나타냅니다. #이진분류 모델이기 때문에, 결과값이 무조건 양성 클래스와 음성 클래스로 해석이 되고. 이때 결과값음 0~1 or 0~(-1) 
                    # 사이의 결과값으로 나온다. 즉 확률인데 보통 이 확률값이 0.5를 넘어서면 양성 클래스 이고 넘지 못하면 음성 클래스 로 분류된다. 
                    
                    
    A, predicted = torch.max(output, 1) # 가장 큰 값뿐만 아니라 해당 값이 나타나는 열의 인덱스 반환합니다.
                                        #torch.max(output, 1) 여기서 1은 최댓값을 찾을 차원 을 의미한다.  output은 텐서 를 의미하고,
                                        #이렇게 할경우, 텐서에서 최댓값을 찾을 건데, 1차원에 존재하는 텐서의 값들 중에서 최댓값을 반환해라 + 그 최댓값의 열의 인덱스 값도 같이 반환을 하게 된다.
                                        #따라서, 최종적으로 torch.max(output, 1)의 출력값은 최댓값 +  그 최댓값의 열의 인덱스를 반환하게 되고 둘다 출력이 되는데 최댓값만 필요하니까 _, predicted를 써준다.
                                        
    # print(A)                            #tensor([0.0747], grad_fn=<MaxBackward0>)
    # print(predicted)                    #temsor([0])          #그 텐서의 최댓값의 열의 인덱스가 반환이 된다!       
    # print(predicted.item())             #0  그 폴더의 인덱스를 뽑아 내기 위해서 사용! 
    percnetage=A.item()                        #0.07945941388607025
    
    return percnetage #predicted.item()는 예측된 클래스의 인덱스를 반환하므로

    
    
# 예시 이미지 경로
image_path = "image_1.png"

percentage = predict_image(image_path)


print(f"Predicted class: {percentage}")

if percentage > 0.05 :
    img_1=cv2.imread('./data/0/20231206_134937.jpg',cv2.IMREAD_COLOR)
    cv2.putText(img_1,f"Predicted class: {percentage:.4f}",(0,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,255),1)
    cv2.imshow('result_0',img_1)
    cv2.waitKey(0)
    

