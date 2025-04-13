import torch
import matplotlib.pyplot as plt
import os

def save_model(model, path):
    """保存模型到指定路径"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """从指定路径加载模型"""
    model.load_state_dict(torch.load(path))
    model.eval()

def plot_loss(losses):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def create_dir_if_not_exists(directory):
    """如果目录不存在，则创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)