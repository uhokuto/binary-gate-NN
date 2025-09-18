import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nn_model
import seaborn as sns
import torch_dataset
from sklearn import preprocessing
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import random
# https://qiita.com/TaigaMasuda/items/24d85860ffcd724de9eb



df = sns.load_dataset("iris")
df = df[(df['species']=='setosa') | (df['species']=='virginica')]
features = df.iloc[:,0:4].values

species = df.iloc[:,4].values
y = [1 if s == 'setosa' else 0 for s in species]
feature_name = df.iloc[:,0:4].columns.tolist()

sc = StandardScaler().fit(features)
features = sc.transform(features)


trainval_data = torch_dataset.create(features,y) # pytorchテンソルデータセット形式に変換
features_sample, label_sample = trainval_data[0]
print(features_sample)
print( label_sample)

train_size = int(len(trainval_data) * 0.75)
test_size = len(trainval_data)-train_size
train_data, test_data = torch.utils.data.random_split(trainval_data, [train_size, test_size])
#print(train_data[0].shape)
BATCH_SIZE = 10
train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,drop_last=True,
                          num_workers=0) 

test_loader = DataLoader(dataset=test_data,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=0)




print("train data size: ",len(train_data))   #train data size:  48000
print("train iteration number: ",len(train_data)//BATCH_SIZE)   #train iteration number:  480
print("val data size: ",len(test_data))   #val data size:  12000
print("val iteration number: ",len(test_data)//BATCH_SIZE)   #val iteration number:  120



input_size = 4
hidden_size = 5
output_size = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn_model.Net(input_size, hidden_size, output_size).to(device)
print(model)

criterion = nn.CrossEntropyLoss() # 損失関数：softmax と　crossentropy誤差を一体化している　また、教師ラベルをone hotにする処理もここで行う
#（実際は、非０に相当する出力値だけ教師ラベルとの差を計算するだけ）

# 最適化法の指定　optimizer：最適化
# SGD：確率的勾配降下法
optimizer = optim.SGD(model.parameters(), lr=0.01)

def run_one_epoch(model, dataloader, criterion, optimizer=None, train=True, device="cpu"):
    if train:
        model.train()  # 訓練モード
    else:
        model.eval()   # 推論モード
    
    running_loss = 0.0
    correct = 0
    total = 0

    # 評価時は勾配を止める
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # 順伝播 outputs はミニバッチ数×出力ノード数の2次元配列
            outputs = model(X)
            # クロスエントロピー誤差の計算
            loss = criterion(outputs, y)

            if train:  # 訓練のときだけ backprop
                optimizer.zero_grad() # 勾配初期化
                loss.backward() # 逆伝播
                optimizer.step() # 勾配の更新

            # ロスの蓄積
            running_loss += loss.item() * X.size(0)

            # 精度の計算
            maxval, predicted = outputs.max(1)#列方向の最大値、インデックスを探す
            correct += (predicted == y).sum().item() # item()は0次元pytorchテンソル型をpythonスカラーに変換する（PyTorch には「純粋なスカラー型」はなく、0次元テンソルで代用している）
            total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc










train_loss_list = []
test_loss_list = []

# epoch数分繰り返す
num_epochs=200
for epoch in range(1, num_epochs+1, 1):

    train_loss,acc = run_one_epoch(model, train_loader, criterion, optimizer,train=True, device=device)
    
    print("epoch : {}, train_loss : {:.5f},accuracy: {:.5f}" .format(epoch, train_loss,acc))

    train_loss_list.append(train_loss)
    
plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

test_loss,acc = run_one_epoch(model, test_loader, criterion, optimizer,train=False, device=device)

print("test_loss : {:.5f},accuracy: {:.5f}" .format(test_loss,acc))




