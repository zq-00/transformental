import torch
import torch.optim as optim

from model import TransformerClassifier
from ttp import data_loaders, test, train


def main():
    device = torch.device('mps')# if torch.backends.mps.is_available() else 'cpu')  # 使用MPS（如果可用）

    # 数据加载
    train_loader, test_loader = data_loaders('data.csv')

    # 初始化模型、损失函数、优化器
    vocab_size = 858  # 根据数据集调整
    model = TransformerClassifier(vocab_size)
    model.to(device)

    criterion = torch.nn.BCELoss()  # 使用BCELoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(50):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy = test(model, test_loader, criterion, device)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}, Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
