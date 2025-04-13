import pytest
import torch
from src.model import MyModel  # 假设模型类名为 MyModel

def test_model_initialization():
    model = MyModel()
    assert model is not None
    assert isinstance(model, MyModel)

def test_model_forward():
    model = MyModel()
    input_tensor = torch.randn(1, 3, 32, 32)  # 假设输入为 (batch_size, channels, height, width)
    output = model(input_tensor)
    assert output is not None
    assert output.shape == (1, 10)  # 假设输出为 (batch_size, num_classes)

def test_model_training():
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    input_tensor = torch.randn(8, 3, 32, 32)  # 假设 batch_size 为 8
    target_tensor = torch.randint(0, 10, (8,))  # 假设有 10 个类

    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    assert loss.item() >= 0  # 确保损失值非负

def test_model_evaluation():
    model = MyModel()
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(input_tensor)
    assert output is not None
    assert output.shape == (1, 10)  # 假设输出为 (batch_size, num_classes)