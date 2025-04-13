import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import SimpleModel
from torchvision import transforms

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # 将数据移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            # 将数据移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

test_files = [
    './data/test_batch'
]

data_files = [
    './data/data_batch_1',
    './data/data_batch_2',
    './data/data_batch_3',
    './data/data_batch_4',
    './data/data_batch_5'
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 模型参数
input_size = 3072  # CIFAR-10 图像的展平大小 (3 * 32 * 32)
hidden_size = 128  # 隐藏层大小
output_size = 10   # CIFAR-10 有 10 个类别

if __name__ == "__main__":
    # 检测设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    train_dataset = CustomDataset(data_files, transform=transform)
    test_dataset = CustomDataset(test_files, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = SimpleModel(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=25)
    
    # 测试模型
    evaluate_model(model, test_loader, device)