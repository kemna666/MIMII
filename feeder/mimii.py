from torch.utils.data import Dataset
import torch

class MIMIIDataset(Dataset):    
    def __init__(self, data):
        self.merged_data =[]
        for snr_data in data:
            self.merged_data.extend(snr_data)
    
    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.merged_data)
    
    def __getitem__(self, idx):
        # 根据索引idx返回一个样本的特征和标签
        mfcc_features = self.merged_data[idx][0][0]
        device_index = torch.tensor(self.merged_data[idx][0][1])
        label_index =   torch.tensor(self.merged_data[idx][1],dtype=torch.long)
       # 将MFCC特征转换为Tensor
        mfcc_features = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
        return mfcc_features,device_index,label_index
