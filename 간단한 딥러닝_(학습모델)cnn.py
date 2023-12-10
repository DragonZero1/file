import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = []
        self.all_labels = []
        for label in os.listdir(main_dir):
            label_dir = os.path.join(main_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    self.all_imgs.append(os.path.join(label_dir, img_file))
                    self.all_labels.append(int(label))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = self.all_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = self.all_labels[idx]
        return tensor_image, label


# CNN 모델 정의
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

# 데이터셋 및 데이터 로더 설정
# 이미지 증강을 위한 변환 추가
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 랜덤으로 이미지를 수평 뒤집기
    transforms.RandomRotation(10),       # 랜덤으로 이미지 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색조 변환
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # 랜덤 크롭 및 사이즈 조정
    transforms.ToTensor()  # PIL 이미지를 텐서로 변환
])
dataset = CustomImageDataset(main_dir="data", transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 모델, 손실 함수, 옵티마이저 초기화
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 학습 루프
num_epochs = 20
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 모델 저장
torch.save(model.state_dict(), "model.pth")


