#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人无关数据处理模块
实现IEMOCAP数据集的说话人无关划分和数据增强
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import random
import re
import os

class SpeakerIndependentDataLoader:
    """说话人无关数据加载器"""
    
    def __init__(self, data_path='./Train_data_org.pickle', test_ratio=0.2, val_ratio=0.1):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
            test_ratio: 测试集比例
            val_ratio: 验证集比例
        """
        self.data_path = data_path
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        
        # IEMOCAP说话人信息
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        self.speakers_per_session = {
            'Session1': ['Ses01F', 'Ses01M'],
            'Session2': ['Ses02F', 'Ses02M'],
            'Session3': ['Ses03F', 'Ses03M'], 
            'Session4': ['Ses04F', 'Ses04M'],
            'Session5': ['Ses05F', 'Ses05M']
        }
        
        # 情感标签映射
        self.emotion_mapping = {
            'ang': 0,  # Angry
            'hap': 1,  # Happy
            'neu': 2,  # Neutral  
            'sad': 3   # Sad
        }
        self.emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
        
        self.data = None
        self.speaker_data = defaultdict(list)
        self.load_data()
        
    def extract_speaker_from_filename(self, filename):
        """从文件名提取说话人信息"""
        # IEMOCAP文件名格式: Ses01F_impro01_F000.wav
        match = re.match(r'(Ses\d+[FM])_', filename)
        if match:
            return match.group(1)
        return None
    
    def load_data(self):
        """加载和预处理数据"""
        print("📁 加载IEMOCAP数据...")
        
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"✅ 成功加载 {len(self.data)} 个样本")
        
        # 按说话人分组数据
        for i, sample in enumerate(self.data):
            # 处理不同的数据格式
            if isinstance(sample, list):
                # 如果是列表格式 [features, label, filename]
                if len(sample) >= 3:
                    filename = sample[2] if isinstance(sample[2], str) else ''
                    features = sample[0]
                    label = sample[1]
                    sample_dict = {'features': features, 'emotion': label, 'id': filename}
                else:
                    print(f"⚠️ 跳过格式不正确的样本 {i}: {type(sample)}")
                    continue
            elif isinstance(sample, dict):
                # 如果已经是字典格式
                filename = sample.get('id', '')
                sample_dict = sample
            else:
                print(f"⚠️ 跳过未知格式的样本 {i}: {type(sample)}")
                continue
                
            speaker = self.extract_speaker_from_filename(filename)
            
            if speaker:
                # 确保情感标签在我们的映射中
                emotion = sample_dict.get('emotion', -1)
                if emotion in [0, 1, 2, 3]:  # 只保留四个主要情感
                    sample_dict['speaker'] = speaker
                    self.speaker_data[speaker].append(sample_dict)
        
        # 统计信息
        print("\n📊 说话人数据分布:")
        total_samples = 0
        emotion_counts = Counter()
        
        for speaker, samples in self.speaker_data.items():
            emotion_dist = Counter([s['emotion'] for s in samples])
            print(f"  {speaker}: {len(samples)} 样本 - {dict(emotion_dist)}")
            total_samples += len(samples)
            emotion_counts.update(emotion_dist)
        
        print(f"\n📈 总体情感分布: {dict(emotion_counts)}")
        print(f"📊 有效样本总数: {total_samples}")
    
    def create_speaker_independent_splits(self, fold_idx=0, n_folds=5):
        """
        创建说话人无关的数据划分
        
        Args:
            fold_idx: 当前折数
            n_folds: 总折数
            
        Returns:
            train_data, val_data, test_data: 训练、验证、测试数据
        """
        print(f"\n🔄 创建第 {fold_idx+1}/{n_folds} 折的说话人无关划分...")
        
        speakers = list(self.speaker_data.keys())
        speakers.sort()  # 确保可重复性
        
        # 使用固定的说话人划分策略
        n_test_speakers = max(1, len(speakers) // n_folds)
        
        # 计算测试说话人
        start_idx = fold_idx * n_test_speakers
        end_idx = min(start_idx + n_test_speakers, len(speakers))
        test_speakers = speakers[start_idx:end_idx]
        
        # 剩余说话人用于训练和验证
        remaining_speakers = [s for s in speakers if s not in test_speakers]
        
        # 从剩余说话人中选择验证说话人
        n_val_speakers = max(1, len(remaining_speakers) // 5)
        val_speakers = remaining_speakers[:n_val_speakers]
        train_speakers = remaining_speakers[n_val_speakers:]
        
        print(f"🎯 训练说话人: {train_speakers}")
        print(f"🎯 验证说话人: {val_speakers}")
        print(f"🎯 测试说话人: {test_speakers}")
        
        # 收集数据
        train_data = []
        val_data = []
        test_data = []
        
        for speaker in train_speakers:
            train_data.extend(self.speaker_data[speaker])
        
        for speaker in val_speakers:
            val_data.extend(self.speaker_data[speaker])
            
        for speaker in test_speakers:
            test_data.extend(self.speaker_data[speaker])
        
        # 数据增强（仅对训练集）
        train_data = self.augment_training_data(train_data)
        
        print(f"📊 数据划分结果:")
        print(f"  训练集: {len(train_data)} 样本")
        print(f"  验证集: {len(val_data)} 样本")
        print(f"  测试集: {len(test_data)} 样本")
        
        return train_data, val_data, test_data
    
    def augment_training_data(self, train_data):
        """
        训练数据增强
        
        Args:
            train_data: 原始训练数据
            
        Returns:
            augmented_data: 增强后的训练数据
        """
        print("🔧 应用训练数据增强...")
        
        augmented_data = train_data.copy()
        
        # 统计各情感类别的样本数
        emotion_counts = Counter([sample['emotion'] for sample in train_data])
        max_count = max(emotion_counts.values())
        
        # 对少数类进行过采样
        for emotion, count in emotion_counts.items():
            if count < max_count * 0.8:  # 如果某类别样本数少于最多类别的80%
                emotion_samples = [s for s in train_data if s['emotion'] == emotion]
                
                # 计算需要增强的数量
                target_count = int(max_count * 0.8)
                need_augment = target_count - count
                
                if need_augment > 0:
                    # 随机选择样本进行增强
                    augment_samples = np.random.choice(emotion_samples, 
                                                     size=min(need_augment, len(emotion_samples)), 
                                                     replace=True)
                    
                    for sample in augment_samples:
                        # 创建增强样本（这里简单复制，实际可以加入噪声等）
                        aug_sample = sample.copy()
                        aug_sample['id'] = aug_sample['id'] + '_aug'
                        augmented_data.append(aug_sample)
        
        # 统计增强后的分布
        final_emotion_counts = Counter([sample['emotion'] for sample in augmented_data])
        print(f"📈 数据增强后情感分布: {dict(final_emotion_counts)}")
        
        return augmented_data
    
    def create_data_loaders(self, train_data, val_data, test_data, batch_size=32, num_workers=4):
        """
        创建PyTorch数据加载器
        
        Args:
            train_data, val_data, test_data: 数据列表
            batch_size: 批次大小
            num_workers: 工作进程数
            
        Returns:
            train_loader, val_loader, test_loader: PyTorch数据加载器
        """
        from torch.utils.data import DataLoader
        
        # 创建数据集
        train_dataset = IEMOCAPDataset(train_data, is_training=True)
        val_dataset = IEMOCAPDataset(val_data, is_training=False)
        test_dataset = IEMOCAPDataset(test_data, is_training=False)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

class IEMOCAPDataset(torch.utils.data.Dataset):
    """IEMOCAP数据集类"""
    
    def __init__(self, data_list, is_training=False, max_length=None):
        """
        初始化数据集
        
        Args:
            data_list: 数据样本列表
            is_training: 是否为训练模式
            max_length: 最大序列长度
        """
        self.data_list = data_list
        self.is_training = is_training
        self.max_length = max_length
        
        # 数据增强参数
        self.noise_factor = 0.005
        self.time_stretch_factor = 0.1
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data_list[idx]
        
        # 获取音频特征
        wav_encodings = sample['wav_encodings']
        if isinstance(wav_encodings, torch.Tensor):
            audio_features = wav_encodings.squeeze()
        else:
            audio_features = torch.tensor(wav_encodings, dtype=torch.float32).squeeze()
        
        # 获取标签和其他信息
        emotion_label = torch.tensor(sample['emotion'], dtype=torch.long)
        sample_id = sample['id']
        speaker = sample.get('speaker', 'unknown')
        
        # 获取说话人标签（用于对抗训练）
        speaker_label = self._get_speaker_label(speaker)
        
        # 训练时应用数据增强
        if self.is_training:
            audio_features = self._apply_augmentation(audio_features)
        
        # 序列长度处理
        seq_length = audio_features.shape[0] if audio_features.dim() > 1 else 1
        
        return {
            'audio_features': audio_features,
            'emotion_label': emotion_label,
            'speaker_label': speaker_label,
            'seq_length': seq_length,
            'sample_id': sample_id,
            'speaker': speaker
        }
    
    def _get_speaker_label(self, speaker):
        """获取说话人标签"""
        speaker_mapping = {
            'Ses01F': 0, 'Ses01M': 1,
            'Ses02F': 2, 'Ses02M': 3,
            'Ses03F': 4, 'Ses03M': 5,
            'Ses04F': 6, 'Ses04M': 7,
            'Ses05F': 8, 'Ses05M': 9
        }
        return torch.tensor(speaker_mapping.get(speaker, 0), dtype=torch.long)
    
    def _apply_augmentation(self, audio_features):
        """应用数据增强"""
        if not self.is_training:
            return audio_features
        
        # 添加高斯噪声
        if random.random() < 0.3:
            noise = torch.randn_like(audio_features) * self.noise_factor
            audio_features = audio_features + noise
        
        # 时间遮蔽（类似SpecAugment）
        if random.random() < 0.2 and audio_features.dim() > 1:
            seq_len = audio_features.shape[0]
            mask_len = int(seq_len * 0.1)  # 遮蔽10%的时间步
            mask_start = random.randint(0, max(0, seq_len - mask_len))
            audio_features[mask_start:mask_start + mask_len] *= 0.1
        
        return audio_features

def collate_fn(batch):
    """自定义批次整理函数"""
    # 提取各个字段
    audio_features = [item['audio_features'] for item in batch]
    emotion_labels = torch.stack([item['emotion_label'] for item in batch])
    speaker_labels = torch.stack([item['speaker_label'] for item in batch])
    seq_lengths = torch.tensor([item['seq_length'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]
    speakers = [item['speaker'] for item in batch]
    
    # 对音频特征进行填充
    if audio_features[0].dim() > 1:
        # 二维特征，需要在序列维度填充
        max_len = max([feat.shape[0] for feat in audio_features])
        padded_features = []
        
        for feat in audio_features:
            if feat.shape[0] < max_len:
                pad_len = max_len - feat.shape[0]
                padded_feat = F.pad(feat, (0, 0, 0, pad_len))
            else:
                padded_feat = feat
            padded_features.append(padded_feat)
        
        audio_features = torch.stack(padded_features)
    else:
        # 一维特征，直接堆叠
        audio_features = torch.stack(audio_features)
    
    return {
        'audio_features': audio_features,
        'emotion_labels': emotion_labels,
        'speaker_labels': speaker_labels,
        'seq_lengths': seq_lengths,
        'sample_ids': sample_ids,
        'speakers': speakers
    }

if __name__ == "__main__":
    # 测试数据加载器
    print("🧪 测试说话人无关数据加载器...")
    
    try:
        loader = SpeakerIndependentDataLoader()
        
        # 测试数据划分
        for fold in range(3):  # 测试前3折
            train_data, val_data, test_data = loader.create_speaker_independent_splits(fold, n_folds=5)
            print(f"✅ 第 {fold+1} 折划分成功")
        
        print("🎉 说话人无关数据加载器测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
