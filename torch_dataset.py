from torch.utils.data import DataLoader,Dataset
import torch



class create(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features_values = features
        self.labels = labels
        

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.features_values)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        
        features_x = torch.FloatTensor(self.features_values[idx])
        labels =  int(self.labels[idx]) 
        #labels = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return features_x, labels
        