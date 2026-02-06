from sklearn.model_selection import train_test_split
import torch
import yaml
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from net.CNN import CNN
from net.DenseNet import DenseNet
from feeder.mimii import MIMIIDataset

optimizer_dict = {'SGD':optim.SGD,'Adam':optim.Adam}
crition_dict = {'CrossEntropyLoss':nn.CrossEntropyLoss}

class Process():
    def __init__(self,file_path):
        with open(file_path,'r') as file:
            config = yaml.safe_load(file)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = self.get_model(config['model']).to(self.device)
            print(f"模型是否在GPU: {next(self.model.parameters()).is_cuda}")  # 应该输出True
            self.optimizer = optimizer_dict[config['optimizer']['optim']](self.model.parameters(),lr=config['optimizer']['lr'])
            self.dataset=MIMIIDataset(config['data']['file_path'])
            self.crition = crition_dict[config['train']['loss']]()
            self.epoches = config['train']['epoches']
            self.batch_size = config['data']['batch_size']
            self.test_size = config['data']['test_size']
            self.train_data, self.test_data = train_test_split(self.dataset, test_size=self.test_size,random_state=config['data']['random_seed'])
            self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
            self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
            
            self.writer = SummaryWriter(
            log_dir=f'./runs/MIMII_{config['model']['model']}-{time.time()}',
            comment=f'_lr{config["optimizer"]["lr"]}_bs{self.batch_size}_ep{self.epoches}'
            )
            
        self.epoch_data = []
        self.test_accuracy = []
        self.train_loss_list = []
        self.train()
        self.accuracy()
       #self.plot()
    def get_model(self,config):
        if config['model'] == 'CNN':
            return CNN(input_dim=config['input_dim'],hidden_dim=config['hidden_dim'],output_dim=config['output_dim'],length=config['length']).to(self.device)
        elif config['model'] == 'DenseNet':
            return DenseNet(input_dim=config['input_dim'],         
            num_init_features=config['num_init_features'],       # 初始通道32（适配128x128输入）
            block_config=config['block_config'],       # 3个DenseBlock，每个4层（轻量版）
            batchnorm_size=config['batchnorm_size'],           # Bottleneck倍数4
            growth_rate=config['growth_rate'],             # 增长率12
            drop_rate=config['drop_rate'],              # Dropout 20%
            compression_rate=config['compression_rate'],       # 压缩率50%
            num_classes=config['num_classes'] ,              # 二分类：正常/异常,
            device = self.device).to(self.device)
        
    def train(self):
        
        for self.epoch in tqdm(range(self.epoches),desc=f'epoch:',unit='epoch'):
            total_train_loss = 0.0
            self.model.train()
            train_loss = 0        
            train_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.epoches}", unit="batch")
            for data in train_bar:
                    device_index = data.y.to(self.device)  # 输入设备索引
                    mfcc_features = data.x.to(self.device)  # 输入MFCC特征
                    output = self.model(mfcc_features)  # 输出标签
                    train_loss = self.crition(output, device_index) 
                    train_loss.backward()
                    self.optimizer.step()
                    total_train_loss += train_loss.item()
            print(f'{self.epoch+1} loss={train_loss.item()}\n')
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.train_loss_list.append(avg_train_loss)
            print(f'Epoch {self.epoch+1}/{self.epoches} | 平均训练损失: {avg_train_loss:.4f}')

            self.writer.add_scalar('Train/Avg_Epoch_Loss', avg_train_loss, global_step=self.epoch+1)
            
            test_acc = self.accuracy()
            self.test_accuracy.append(test_acc)
            self.writer.add_scalar('Test/Epoch_Accuracy', test_acc, global_step=self.epoch+1)
                
    def accuracy(self):
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                test_bar = tqdm(self.test_loader, desc=f"Epoch {self.epoch+1}/{self.epoches} test", unit="batch")
                for data in test_bar:
                    device_index = data.y.to(self.device)  # 输入设备索引
                    mfcc_features = data.x.to(self.device)  # 输入MFCC特征
                    output = self.model(mfcc_features,mode='train')  # 输出标签
                    _, predicted = torch.max(output.data, 1)
                    total += device_index.size(0)
                    correct += (predicted == device_index).sum().item()
            acc = correct / total
            print(f'Accuracy: {acc:.4f}')
            self.epoch_data.append(self.epoch)
            self.test_accuracy.append(acc)
            return acc

    def collate_fn(batch):
        batch = [data for data in batch if data is not None]
        return Batch.from_data_list(batch)



# 测试使用示例
if __name__ == "__main__":
    Process('config/densenet.yaml')