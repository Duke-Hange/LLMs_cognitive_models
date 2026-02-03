import pandas as pd
import torch
import torch.nn as nn
import ast
from utils import DataProcessor, train
from models import MoT, ContextNN
from sklearn.model_selection import train_test_split    
import numpy as np
from os.path import join
import os
import gc


if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')


data = pd.read_csv('data.csv')
data['A'] = data['A'].apply(ast.literal_eval)
data['B'] = data['B'].apply(ast.literal_eval)

dataset = DataProcessor(data)
X, y= dataset.get_darray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1017)


splits = 50

X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

mot_performances = []
context_performances = []
save_path = join(os.getcwd(), 'results')
os.makedirs(save_path, exist_ok=True)

for i in range(1, splits+1):
    print(f'Training on {i/splits*100}% of the data')
    end_idx = int(len(X_train)*i/splits)
    X_train_percent = X_train[:end_idx]
    y_train_percent = y_train[:end_idx]

    mot = MoT().to(device)
    context_model = ContextNN().to(device)
    
    # 训练模型
    trained_mot, mot_history = train(mot, X_train_percent, y_train_percent, device, 
                                    epochs=2000, batch_size=64, verbose=False, lr=0.0001)

    trained_context, context_history = train(context_model, X_train_percent, y_train_percent, device, 
                                            epochs=2000, batch_size=64, verbose=False, lr=0.0001)
    # 记录模型性能
    trained_mot.eval()
    with torch.no_grad():
        y_pred = trained_mot(X_test)
        mse = nn.MSELoss()(y_pred, y_test)
        mot_performances.append(mse.item())
    
    trained_context.eval()
    with torch.no_grad():
        y_pred = trained_context(X_test)
        mse = nn.MSELoss()(y_pred, y_test)
        context_performances.append(mse.item())

    # 保存模型
    torch.save(trained_mot, join(save_path, f'mot_model{i}.pth'))
    torch.save(trained_context, join(save_path, f'context_model{i}.pth'))
    torch.save(mot_history, join(save_path, f'mot_history{i}.pth'))
    torch.save(context_history, join(save_path, f'context_history{i}.pth'))

    del mot, context_model, trained_mot, trained_context
    del mot_history, context_history
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 每10次迭代强制垃圾回收
    if i % 10 == 0:
        gc.collect()

np.save(join(save_path, 'mot_performances.npy'), mot_performances)
np.save(join(save_path, 'context_performances.npy'), context_performances)

