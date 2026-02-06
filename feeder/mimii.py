from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import pickle
class MIMIIDataset(Dataset):    
    def __init__(self, pkl_file_path):
        # 加载.pkl文件中的数据
        with open(pkl_file_path, 'rb') as file:
            self.data = pickle.load(file)
        self.merged_data =[]
        for snr_data in self.data:
            self.merged_data.extend(snr_data)
    
    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.merged_data)
    
    def __getitem__(self, idx):
        # 根据索引idx返回一个样本的特征和标签
        mfcc_features = self.merged_data[idx][0][0]
        device_index = self.merged_data[idx][0][1]
        label_index =   self.merged_data[idx][1]
       # 将MFCC特征转换为Tensor
        mfcc_features = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
        return Data(
                    x=mfcc_features.detach().clone() if isinstance(mfcc_features, torch.Tensor) else torch.tensor(mfcc_features),
                    edge_index=label_index,
                    y=torch.tensor(device_index, dtype=torch.long).detach().clone()
                )
