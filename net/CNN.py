import torch.nn as nn 
import torch

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, length):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.Conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=1, padding=1)
        self.Conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        conv1_out_length = (length + 2*1 - 7) // 1 + 1
        pool1_out_length = (conv1_out_length - 3) // 2 + 1
        conv2_out_length = (pool1_out_length + 2*1 - 3) // 1 + 1
        pool2_out_length = (conv2_out_length - 3) // 2 + 1
        self.fc = nn.Linear(output_dim * pool2_out_length, 13)  # 输出维度为13

    def forward(self, x,mode='test'):
        x = self.Conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        if mode == 'train':
            x = self.dropout(x)
        x = self.Conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        if mode == "train":
            x = self.dropout(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # 只返回标签
    