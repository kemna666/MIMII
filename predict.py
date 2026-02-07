import os
import torch 
import yaml
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
# 注意：这里需要根据你的实际路径导入模型和数据集类
from net.CNN import CNN
from net.DenseNet import DenseNet
from feeder.mimii import MIMIIDataset


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
            checkpoint_path = f'{self.model_save_dir}/{"final" if os.path.exists(f"{self.model_save_dir}/final_model.pth") else "best"}_model.pth'
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
            # ========== 关键修改1 ==========
            # 确保维度是 [1, n_mfcc, time_steps] (通道数=1)
            # 原代码: mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # 添加通道维度 [1, 13, time_steps]
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
        # 获取所有音频路径
        audio_paths = self._get_audio_files(input_path)
        batch_size = batch_size or self.batch_size
        self.model.eval()

        # 1. 预处理所有音频
        features_list = []
        for audio_path in tqdm(audio_paths, desc="预处理音频"):
            mfcc = self.preprocess_audio(audio_path)
            features_list.append(mfcc)

        # ========== 关键修改2 ==========
        # 原代码: all_features = torch.cat(features_list, dim=0).to(self.device)
        # 修正：确保拼接后维度为 [batch_size, 1, n_mfcc, time_steps]
        all_features = torch.cat(features_list, dim=0).to(self.device)
        # 额外检查并修正通道维度（防止意外的维度错误）
        if all_features.dim() == 3:
            all_features = all_features.unsqueeze(1)  # 添加通道维度
        elif all_features.size(1) != 1:
            # 如果通道数不是1，取第一个通道（兼容处理）
            all_features = all_features[:, 0:1, :, :]
        
        total = len(all_features)
        num_batches = (total + batch_size - 1) // batch_size  # 向上取整

        results = []
        with torch.no_grad():
            for b in tqdm(range(num_batches), desc="批量推理", unit="batch"):
                # 取当前批次
                start = b * batch_size
                end = min(start + batch_size, total)
                batch_feat = all_features[start:end]

                # 推理
                outputs = self.model(batch_feat)
                _, preds = torch.max(outputs, 1)
                confs = torch.softmax(outputs, dim=1)[range(len(preds)), preds].cpu().numpy()

                # 组装结果
                for i_in_batch, (p, c) in enumerate(zip(preds.cpu().numpy(), confs)):
                    idx = start + i_in_batch
                    result = {
                        'audio_path': audio_paths[idx],
                        'device_idx': device_idx,
                        'predicted_label_idx': int(p),
                        'confidence': round(float(c), 4),
                        'predicted_label_name': class_names[int(p)] if class_names else None
                    }
                    results.append(result)

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