import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa
import pickle
#引入线程池
from concurrent.futures import ThreadPoolExecutor,as_completed

snrs = ['-6db','0db','6db']
devices = ['fan','pump','slider','valve']
ids = ['id_00', 'id_02', 'id_04', 'id_06']
labels = ['abnormal', 'normal']

# 定义映射字典
snr_map = {'-6db': 1, '0db': 2, '6db': 3}
device_map = {'fan': 1, 'pump': 2, 'slider': 3, 'valve': 4}
label_map = {'abnormal': -1, 'normal': 1}
    

#获取文件列表
def get_filelist():
    file_list = []
    for root,dirs,files in os.walk('data'):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.relpath(root,'data')
                parts = path.split(os.sep)
                if len(parts)==4:
                    snr, device, _, label = parts
                    file_list.append((
                       os.path.join(root,file),
                       snr_map[snr],
                       device_map[device],
                       label_map[label]
                    ))
    return file_list

def process_file(file):
    try:
        audio_data ,_= sf.read(file[0],always_2d=True) 
        audio = audio_data.transpose(1, 0)[0].astype(np.float32)
        label = [file[1], file[2], file[3]]
        mfcc = librosa.feature.mfcc(
                y=audio,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )
        return mfcc.astype(np.float32),label
    except Exception as e:
        print(f'[Error]{os.path.basename(file[0])}:{e}')

def extract_data(file_list,batch_size):
    total = len(file_list)
    mfcc_shape = (total,13,313)
    label_shape = (total, 3)

    mfcc_mmap = np.lib.format.open_memmap(
        'data/mfcc_mmap.npy', dtype=np.float32, mode='w+', shape=mfcc_shape
    )
    label_mmap = np.lib.format.open_memmap(
        'data/label_mmap.npy', dtype=np.int8, mode='w+', shape=label_shape
    )
    pbar = tqdm(total=total,desc="处理wav文件",unit="units")
    with ThreadPoolExecutor(max_workers=32) as executor:
        for i in range(0, total, batch_size):
            batch = file_list[i:i+batch_size]
            futures = [executor.submit(process_file, info) for info in batch]

            for j, future in enumerate(as_completed(futures)):
                mfcc, label = future.result()
                if mfcc is not None and label is not None:
                    idx = i + j
                    mfcc_mmap[idx] = mfcc
                    label_mmap[idx] = label
                pbar.update(1)

    pbar.close()
    # 音频数组：(样本数, 160000)，float32
    mfcc_mmap.flush()
    label_mmap.flush()
    print(f"[Success] 原始音频数组维度：{mfcc_mmap.shape}")
    print(f"[Success] 标签数组维度：{label_mmap.shape}")
    return mfcc_mmap,label_mmap

def generate_pkl(mfcc_data,label_data):

    assert len(mfcc_data) == len(label_data), "MFCC和标签样本数不一致！"
    
    # 按SNR分组初始化字典（key=SNR值，value=该SNR的所有样本）
    snr_grouped_data = {snr_val: [] for snr_val in snr_map.values()}  # key:1/-6db, 2/0db,3/6db
    
    # 遍历所有样本，按要求格式化并分组
    pbar = tqdm(total=len(mfcc_data), desc="重组样本", unit="sample")
    for mfcc, label in zip(mfcc_data, label_data):
        snr_val = label[0]   
        device_val = label[1]
        label_val = label[2] 
        sample = ((mfcc, device_val), label_val)
        # 将样本加入对应SNR的分组
        snr_grouped_data[snr_val].append(sample)
        pbar.update(1)
    pbar.close()
    
    final_data = [
        snr_grouped_data[1],  # -6db的所有样本
        snr_grouped_data[2],  # 0db的所有样本
        snr_grouped_data[3]   # 6db的所有样本
    ]
    

    with open('data.pkl', 'wb') as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    for i, snr_name in enumerate(snrs):
        print(f"[Success] {snr_name} 样本数：{len(final_data[i])}")
    print(f"[Success] pkl文件已保存，总样本数：{sum(len(x) for x in final_data)}")


if __name__ == '__main__':
    mfcc_data,label_data = extract_data(get_filelist(),1024)
    generate_pkl(mfcc_data,label_data)
    print('成功')

