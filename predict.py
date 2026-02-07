import os
import torch 
import yaml
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from .net import CNN,DenseNet
from .feeder.mimii import MIMIIDataset


class AudioPredictor:
    def __init__(self, config_path, input_path=None):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = self.config['data'].get('batch_size', 8)
        self.model_save_dir = self.config['model_save_dir']
        self.n_mfcc = 13 # 与预处理保持一致
        self.sr = 16000     # 采样率
        self.model = self.get_model(self.config['model']).to(self.device)
        self._load_weights()
        
        if input_path is not None:
            self.results = self.infer_batch(input_path)

    def get_model(self,model_config):
        if model_config['model'] == 'CNN':
            return CNN(input_dim=model_config['input_dim'],
                      hidden_dim=model_config['hidden_dim'],
                      output_dim=model_config['output_dim'],
                      length=model_config['length']).to(self.device)
        elif model_config['model'] == 'DenseNet':
            return DenseNet(input_dim=model_config['input_dim'],         
            num_init_features=model_config['num_init_features'],       
            block_config=model_config['block_config'],       
            batchnorm_size=model_config['batchnorm_size'],           
            growth_rate=model_config['growth_rate'],             
            drop_rate=model_config['drop_rate'],              
            compression_rate=model_config['compression_rate'],       
            num_classes=model_config['num_classes'] ,              
            device = self.device).to(self.device)
            
    def _load_weights(self):
        try:
            checkpoint_path = f'{self.model_save_dir}/final_model.pth'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # 处理两种常见的权重保存格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"成功加载模型权重: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def preprocess_audio(self, audio_path):
        try:
            y, _ = librosa.load(audio_path, sr=self.sr)
            
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )
            # 标准化
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            return mfcc_tensor
        except Exception as e:
            raise RuntimeError(f"音频预处理失败 {audio_path}: {str(e)}")

    def _get_audio_files(self, input_path):
        """辅助函数：获取指定路径下的所有音频文件"""
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a')  # 支持的音频格式
        audio_paths = []
        
        # 如果是文件，直接返回包含该文件的列表
        if os.path.isfile(input_path) and input_path.lower().endswith(audio_extensions):
            audio_paths = [input_path]
        # 如果是目录，遍历目录下所有音频文件
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(audio_extensions):
                        audio_paths.append(os.path.join(root, file))
        
        if not audio_paths:
            raise ValueError(f"在路径 {input_path} 中未找到任何音频文件")
        
        return audio_paths
    
    def save_results_to_txt(self, results, save_path='inference_results.txt'):
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                # 写入头部信息
                f.write(f"音频推理结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                # 写入每条结果
                for i, res in enumerate(results, 1):
                    f.write(f"[{i}] 文件路径: {res['audio_path']}\n")
                    f.write(f"    设备索引: {res['device_idx']}\n")
                    f.write(f"    预测标签索引: {res['predicted_label_idx']}\n")
                    f.write(f"    预测标签名称: {res['predicted_label_name']}\n")
                    f.write(f"    置信度: {res['confidence']}\n")
                    f.write("-"*80 + "\n")
            
            print(f"结果已成功保存到: {os.path.abspath(save_path)}")
            return True
        except Exception as e:
            print(f"保存结果失败: {str(e)}")
            return False

    def infer_batch(self, input_path, device_idx=1, class_names=None, batch_size=None):
        # 获取所有音频文件路径
        audio_paths = self._get_audio_files(input_path)
        
        batch_size = batch_size or self.batch_size
        self.model.eval()  

        features_list = []
        for audio_path in tqdm(audio_paths, desc="预处理音频"):
            mfcc = self.preprocess_audio(audio_path)
            features_list.append(mfcc)
        

        mfcc_tensor = torch.cat(features_list, dim=0)
        dev_idx_tensor = torch.tensor([device_idx] * len(audio_paths), dtype=torch.int64)
        dataset = MIMIIDataset(mfcc_tensor, dev_idx_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 3. 批量推理
        results = []
        idx = 0
        with torch.no_grad():  
            for mfcc_features, dev_idx in tqdm(dataloader, desc="批量推理", unit="batch"):
                mfcc_features = mfcc_features.to(self.device)
                dev_idx = dev_idx.to(self.device)

                outputs = self.model(mfcc_features) 
                _, predictions = torch.max(outputs.data, 1)
                confidences = torch.softmax(outputs, dim=1)[range(len(predictions)), predictions].cpu().numpy()
                
                for pred, conf in zip(predictions.cpu().numpy(), confidences):
                    result = {
                        'audio_path': audio_paths[idx],
                        'device_idx': int(dev_idx[idx % len(dev_idx)].cpu().numpy()),
                        'predicted_label_idx': int(pred),
                        'confidence': round(float(conf), 4),
                        'predicted_label_name': class_names[int(pred)] if class_names else None
                    }
                    results.append(result)
                    idx += 1
        
        return results


if __name__ == '__main__':
    # 配置文件路径
    CONFIG_PATH = 'config/densenet.yaml'
    INPUT_PATH = 'data/test/'  
    # 结果保存路径
    SAVE_PATH = 'inference_results.txt'
    
    CLASS_NAMES = ['abnormal', 'normal']
    
    predictor = AudioPredictor(CONFIG_PATH) 
    # 执行批量推理
    results = predictor.infer_batch(
        input_path=INPUT_PATH,
        device_idx=1, 
        class_names=CLASS_NAMES,
        batch_size=4
    )
    
    predictor.save_results_to_txt(results, SAVE_PATH)
    