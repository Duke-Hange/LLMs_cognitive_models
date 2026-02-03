import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from IPython.display import display, clear_output
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold

class DataProcessor:
    def __init__(self, data=None):
        self.data = data
    
    def fix_dim(self):
        data = self.data
        fixed_dim = max(len(row['B']) for _, row in data.iterrows())
        return fixed_dim

    def get_darray(self):
        data = self.data
        fixed_dim = self.fix_dim()
        n_problems = len(data)
        X = np.zeros((n_problems, 4, fixed_dim))

        for i, row in data.iterrows():
            # 处理选项A
            items_a = row['A']
            if items_a:
                items_a = np.array(items_a)
                # 根据实际数据格式调整索引
                X[i, 0, :len(items_a)] = items_a[:, 1]  # 结果值（第二列）
                X[i, 1, :len(items_a)] = items_a[:, 0]  # 概率（第一列）
            
            # 处理选项B
            items_b = row['B']
            if items_b:
                items_b = np.array(items_b)
                # 根据实际数据格式调整索引
                X[i, 2, :len(items_b)] = items_b[:, 1]  # 结果值（第二列）
                X[i, 3, :len(items_b)] = items_b[:, 0]  # 概率（第一列）
        
        y = 1 - data['bRate'].values
        return X, y

def render(history):
    clear_output(wait=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    train_loss = history['train_loss']
    epochs = range(1, len(train_loss)+1)

    ax.plot(epochs, train_loss, label='Training Loss', color='#ac6f82', linewidth=6, alpha=0.8)
    ax.set_xlabel('Epochs', fontsize=17, fontweight='bold')
    ax.set_ylabel('MSE Loss', fontsize=17, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'))

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(2.5)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.tight_layout()
    display(fig)
    plt.close(fig)




def train(model, X_train, y_train, X_test, y_test, device, epochs=100, batch_size=32, verbose=True, lr=0.001):
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # 训练过程记录
    history = {'train_loss': [], 'test_loss': []}
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            test_loss = criterion(y_pred_test, y_test)
            history['test_loss'].append(test_loss.item())

        # 更新学习率
        scheduler.step(avg_train_loss)

        if verbose and (epoch+1) % 10 == 0:
            render(history)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss.item():.4f}')

    return model, history