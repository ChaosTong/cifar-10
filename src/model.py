import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # 输入通道为3（RGB），输出通道为6
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)  # 第二个卷积层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # 全连接层1
        self.fc2 = nn.Linear(120, 84)  # 全连接层2
        self.fc3 = nn.Linear(84, num_classes)  # 输出层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积1 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积2 -> ReLU -> 池化
        x = x.view(-1, 16 * 7 * 7)  # 展平操作
        x = F.relu(self.fc1(x))  # 全连接层1 -> ReLU
        x = F.relu(self.fc2(x))  # 全连接层2 -> ReLU
        x = self.fc3(x)  # 输出层
        return x

    def train_model(self, train_loader, criterion, optimizer, num_epochs, device, test_loader=None):
        for epoch in range(num_epochs):
            self.train()  # 确保模型处于训练模式
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            # 每 10 个 epoch 评估一次
            if (epoch + 1) % 10 == 0 and test_loader is not None:
                print(f"Evaluating at epoch {epoch+1}...")
                self.eval()  # 确保模型处于评估模式
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self.forward(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print(f'Accuracy at epoch {epoch+1}: {accuracy:.2f}%')