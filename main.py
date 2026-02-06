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
from net.net_dict import model_dict
from feeder.mimii import MIMIIDataset

optimizer_dict = {'SGD':optim.SGD,'Adam':optim.Adam}
crition_dict = {'CrossEntropyLoss':nn.CrossEntropyLoss}

class Process():
    def __init__(self,file_path):
        with open(file_path,'r') as file:
            config = yaml.safe_load(file)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = model_dict[config['model']['model']](input_dim=13,hidden_dim=128,output_dim=13,length=313).to(self.device)
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
            log_dir=f'./runs/MIMII_CNN-{time.time()}',
            comment=f'_lr{config["optimizer"]["lr"]}_bs{self.batch_size}_ep{self.epoches}'
            )
            
        self.epoch_data = []
        self.test_accuracy = []
        self.train_loss_list = []
        self.train()
        self.accuracy()
       #self.plot()

    def train(self):
        
        for self.epoch in tqdm(range(self.epoches),desc=f'epoch:',unit='epoch'):
            total_train_loss = 0.0
            self.model.train()
            train_loss = 0        
            train_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.epoches}", unit="batch")
            for data in train_bar:
                    device_index = data.y.to(self.device)  # 输入设备索引
                    mfcc_features = data.x.to(self.device)  # 输入MFCC特征
                    output = self.model(mfcc_features, device_index)  # 输出标签
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
                    output = self.model(mfcc_features, device_index,mode='train')  # 输出标签
                    _, predicted = torch.max(output.data, 1)
                    total += device_index.size(0)
                    correct += (predicted == device_index).sum().item()
            acc = correct / total
            print(f'Accuracy: {acc:.4f}')
            self.epoch_data.append(self.epoch)
            self.test_accuracy.append(acc)
            return acc

    def plot(self):
        # 绘制训练损失和测试精度曲线（双坐标轴，更直观）
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 左轴：训练损失
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Train Loss', color='red', fontsize=12)
        ax1.plot(self.epoch_data, self.train_loss_list, 'r-', label='Train Loss')
       # ax1.tick_params(axis='y', labelcolor='red')
        ax1.legend(loc='upper left')
        
        # 右轴：测试精度
        ax2 = ax1.twinx()
        ax2.set_ylabel('Test Accuracy', color='blue', fontsize=12)
        ax2.plot(self.epoch_data, self.test_accuracy, 'b-', label='Test Accuracy')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.legend(loc='upper right')
        
        # 保存并显示图片
        plt.title('Train Loss and Test Accuracy', fontsize=14)
        plt.tight_layout()  # 防止标签重叠
        save_path = f'./output/accuracy_loss_{int(time.time())}.png'
        plt.savefig(save_path)
        print(f'训练曲线已保存至：{save_path}')
        plt.show()

    def collate_fn(batch):
        batch = [data for data in batch if data is not None]
        return Batch.from_data_list(batch)



if __name__=='__main__':
    Process('config/config.yaml')