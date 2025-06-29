import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def prepare_data(file_path):
    # 读取文件数据
    with open(file_path, 'r') as f:
        lines = f.readlines()

    features = []
    labels = []

    for line in lines:
        # 分割每行数据，按 | 或 ｜ 分隔
        parts = line.strip().split('|,|')
        if len(parts) != 3:
            continue  # 如果该行数据格式不符合预期，跳过

        # 特征部分：以空格分隔的数字
        feature = list(map(int, parts[1].strip().split()))
        feature.extend([-1] * (104 - len(feature)))

        # 标签部分：以空格分隔的标签索引，转为one-hot编码
        label = parts[2].strip().split()
        one_hot = [0] * 17
        for idx in label:
            one_hot[int(idx) - 1] = 1

        features.append(feature)
        labels.append(one_hot)

    # 转换为Tensor
    X = torch.tensor(features, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float)

    # 数据集划分
    return train_test_split(X, y, test_size=0.3)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 计算准确率
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy


def create_data_loaders(file_path, batch_size=32):
    X_train, X_test, y_train, y_test = prepare_data(file_path)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
