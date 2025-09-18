import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size,  output_size,l=-0.1, r=1.1):
        super().__init__()
        
         
        self.fc1 = nn.Linear(input_size, hidden_size)        
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        
        self.alpha = nn.Parameter(torch.full((input_size,), -2.0))
        self.beta  = nn.Parameter(torch.full((input_size,), 0.5))
        
        self.register_buffer("l", torch.tensor(float(l)))
        self.register_buffer("r", torch.tensor(float(r)))
        self.register_buffer("eps", torch.tensor(1e-8))

    def binary_gate(self):
        
        u = torch.rand_like(self.alpha)
                                
        # 正値に投影（数値安定のため小さな eps を足すのも可）
        alpha_pos = F.softplus(self.alpha) + self.eps
        beta_pos  = F.softplus(self.beta)  + self.eps
        s=torch.sigmoid((torch.log(u+ self.eps) - torch.log(1-u+ self.eps) + torch.log(alpha_pos))/beta_pos)
        s_ = s*(self.r-self.l)+self.l
        z  = torch.clamp(s_, 0.0, 1.0)
        #out = torch.maximum(s_, torch.tensor(0.0))
        #z = torch.minimum(out, torch.tensor(1.0))
        return z
        
    def _expected_z(self):
        """推論時: 期待値で z を決める（u を使わないバージョン）"""
        
        
        alpha_pos = F.softplus(self.alpha) + self.eps
        beta_pos  = F.softplus(self.beta)  + self.eps
        s  = torch.sigmoid(torch.log(alpha_pos) / beta_pos)
        s_ = s * (self.r - self.l) + self.l
        z  = torch.clamp(s_, 0.0, 1.0)
        return z

    def forward(self, x): # x : 入力
        
          
        
        if self.training:
            z = self.binary_gate()
        else:
            z = self._expected_z()
        #z = self.binary_gate()
        # x @ W.T + b
        #W_sparce = z * self.fc1.weight # 最適化後にW_sparceを取り出せば、どれくらいL0正則化が効いているかがわかる
        #yin = x @ W_sparce.T + self.fc1.bias#linear.weight は (out_features, in_features) の形なので、入力 x (N, in_features) と掛けるには 転置 W.T が必要
        #print(self.fc1.weight.shape)
        x_gated = x * z
        yin = self.fc1(x_gated)
        #yin = self.fc1(x)
        yout = torch.sigmoid(yin)        
        zin = self.fc2(yout)
        return zin,z
        
    def L0(self):
        alpha_pos = F.softplus(self.alpha) + self.eps
        beta_pos  = F.softplus(self.beta)  + self.eps
        l0_penalty = torch.sigmoid(torch.log(alpha_pos) - beta_pos * torch.log(-self.l / self.r))
        return l0_penalty.sum()