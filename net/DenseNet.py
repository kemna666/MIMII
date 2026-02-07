from typing import OrderedDict
import torch
import torch.nn as nn

class _DenseLayer(nn.Module):
    def __init__(self,input_features,growth_rate,batchNorm_size,drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_features,batchNorm_size*growth_rate,kernel_size=1,stride=1)
        self.norm2 = nn.BatchNorm2d(batchNorm_size*growth_rate)
        self.conv2 = nn.Conv2d(batchNorm_size*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self,x):
        new_features = self.norm1(x)
        new_features = self.relu(new_features)
        new_features = self.conv1(new_features)
        new_features = self.norm2(new_features)
        new_features = self.relu(new_features)
        new_features = self.conv2(new_features)
        if self.drop_rate>0:
            new_features = self.dropout(new_features)
        return torch.cat([x,new_features],dim=1)
    
class _DenseBlock(nn.ModuleList):
    def __init__(self,num_layers,input_features,growth_rate,batchnorm_size,drop_rate):
        super(_DenseBlock,self).__init__()
        self.layers = []
        for i in range(num_layers):
            layer = _DenseLayer(input_features+i*growth_rate,growth_rate=growth_rate,batchNorm_size=batchnorm_size,drop_rate=drop_rate)
            self.add_module(f"layer_{i}", layer)
        self.num_layers = num_layers

    def forward(self,x):
        for i in range(self.num_layers):
            layer = getattr(self, f"layer_{i}")
            x = layer(x)
        return x
    
class _Transition(nn.Module):
    def __init__(self, input_feature,output_feature):
        super().__init__()
        self.norm = nn.BatchNorm2d(input_feature)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(input_feature,output_feature,kernel_size=1,stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2)
    
    def forward(self,x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
    
class DenseNet(nn.Module):
    def __init__(self,input_dim,num_init_features,block_config,batchnorm_size,growth_rate,drop_rate,compression_rate,num_classes,device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(input_dim,num_init_features,kernel_size=7,stride=2,padding=3,bias=False)),
            ('norm0',nn.BatchNorm2d(num_init_features)),
            ('relu0',nn.ReLU(inplace=True)),
            ('pool0',nn.MaxPool2d(3,stride=2,padding=1))
        ])).to(device)

        num_features = num_init_features
        for i,num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers,num_features,growth_rate,batchnorm_size,drop_rate)
            self.features.add_module(f'block-{i+1}',block)
            num_features += num_layers*growth_rate
            if i != len(block_config)-1:
                transition = _Transition(num_features,int(num_features*compression_rate))
                self.features.add_module(f'transition_{i+1}',transition)
                num_features = int(num_features*compression_rate)
        
        self.features.add_module('norm5',nn.BatchNorm2d(num_features))
        self.features.add_module('relu5',nn.ReLU(inplace=True))

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Linear(num_features, num_classes)
        
        for m in self.modules():
            #kaiming Norm
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            # initize weight&bias
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.bias,0)
                nn.init.constant_(m.weight,1)
            elif isinstance(m,nn.Linear):
                nn.init.constant_(m.bias,0)
    
    def forward(self,x,mode='test'):
        features = self.features(x)

        out = self.global_avg_pool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out