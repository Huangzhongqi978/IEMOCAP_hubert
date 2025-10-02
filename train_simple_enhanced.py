#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的增强IEMOCAP语音情感识别训练系统
专注于GRU架构改进，不使用复杂的说话人无关划分
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import argparse
from datetime import datetime
import warnings
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from models.enhanced_gru import create_enhanced_model
from utils import *

warnings.filterwarnings('ignore')

class SimpleEnhancedTrainer:
    """简化的增强训练器"""
    
    def __init__(self, args):
        """初始化训练器"""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
        # 情感标签
        self.emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
        self.emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']
        
        # 创建实验目录
        self.exp_name = f"simple_enhanced_emotion_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join('experiments', self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        print(f"🚀 简化增强训练器初始化完成")
        print(f"📁 实验目录: {self.exp_dir}")
        print(f"🖥️  设备: {self.device}")
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载和预处理数据 - 仿照train.py的逻辑"""
        print("📁 加载IEMOCAP数据...")
        
        with open(self.args.data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 成功加载 {len(data)} 个会话")
        
        # 展开所有会话的数据
        all_samples = []
        for session_idx, session_data in enumerate(data):
            print(f"  会话 {session_idx}: {len(session_data)} 个样本")
            all_samples.extend(session_data)
        
        print(f"📊 总样本数: {len(all_samples)}")
        
        # 提取特征和标签 - 仿照utils.py中的Feature函数
        features = []
        labels = []
        valid_samples = 0
        
        for i, sample in enumerate(all_samples):
            try:
                # 按照原始数据格式处理
                if isinstance(sample, dict) and 'wav_encodings' in sample and 'label' in sample:
                    feature = sample['wav_encodings']
                    label = sample['label']
                    
                    # 转换为tensor
                    if not isinstance(feature, torch.Tensor):
                        feature = torch.tensor(feature, dtype=torch.float32)
                    
                    # 确保特征维度正确 [seq_len, feature_dim]
                    if feature.dim() == 1:
                        feature = feature.unsqueeze(-1)  # [seq_len] -> [seq_len, 1]
                    elif feature.dim() > 2:
                        feature = feature.view(-1, feature.size(-1))  # flatten to [seq_len, feature_dim]
                    
                    # 确保标签有效
                    if isinstance(label, (int, float)) and 0 <= label <= 3:
                        features.append(feature)
                        labels.append(int(label))
                        valid_samples += 1
                    else:
                        print(f"⚠️ 跳过无效标签样本 {i}: label={label}")
                else:
                    print(f"⚠️ 跳过格式错误样本 {i}: {type(sample)}")
                    
            except Exception as e:
                print(f"⚠️ 处理样本 {i} 时出错: {e}")
                continue
        
        print(f"📊 有效样本数: {valid_samples}")
        
        if valid_samples == 0:
            raise ValueError("没有有效的训练样本！")
        
        # 转换为numpy数组用于划分
        self.features = features
        self.labels = labels
        
        # 统计标签分布
        label_counts = np.bincount(labels, minlength=4)
        print("📈 情感分布:")
        for i, count in enumerate(label_counts):
            print(f"  {self.emotion_labels[i]}: {count} 样本")
    
    def create_data_loaders(self):
        """创建数据加载器"""
        print("🔄 创建数据加载器...")
        
        # 简单的训练/测试划分
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.features, self.labels, test_size=0.3, random_state=self.args.seed, 
            stratify=self.labels if len(set(self.labels)) > 1 else None
        )
        
        # 从训练集中再划分出验证集
        if len(train_features) > 2:
            train_features, val_features, train_labels, val_labels = train_test_split(
                train_features, train_labels, test_size=0.2, random_state=self.args.seed,
                stratify=train_labels if len(set(train_labels)) > 1 else None
            )
        else:
            val_features, val_labels = test_features, test_labels
        
        # 填充序列到相同长度
        def pad_sequences(features):
            if len(features) == 0:
                return torch.empty(0, 0, 0)
            
            max_len = max(f.size(0) for f in features)
            feature_dim = features[0].size(1) if features[0].dim() > 1 else features[0].size(0)
            
            padded = []
            for feat in features:
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                
                if feat.size(0) < max_len:
                    pad_len = max_len - feat.size(0)
                    feat = F.pad(feat, (0, 0, 0, pad_len))
                
                padded.append(feat)
            
            return torch.stack(padded)
        
        # 填充特征
        train_features_padded = pad_sequences(train_features)
        val_features_padded = pad_sequences(val_features)
        test_features_padded = pad_sequences(test_features)
        
        # 转换标签为tensor
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_labels = torch.tensor(val_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        
        # 创建数据集
        train_dataset = TensorDataset(train_features_padded, train_labels)
        val_dataset = TensorDataset(val_features_padded, val_labels)
        test_dataset = TensorDataset(test_features_padded, test_labels)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        
        print(f"📊 数据划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def train_model(self):
        """训练模型"""
        print("\n🚀 开始训练增强GRU模型...")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # 创建模型 - 只使用GRU部分，不使用HuBERT
        from models.enhanced_gru import EnhancedGRUModel
        
        # 获取特征维度
        sample_feature = self.features[0]
        input_dim = sample_feature.size(-1)
        
        model = EnhancedGRUModel(
            input_size=input_dim,
            hidden_size=self.args.hidden_layer,
            output_size=self.args.out_class,
            args=self.args
        )
        model = model.to(self.device)
        
        print(f"🎯 输入特征维度: {input_dim}")
        print(f"🎯 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 定义优化器和调度器
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.args.T_0)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        train_history = {'loss': [], 'acc': []}
        val_history = {'loss': [], 'acc': []}
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        print("🔄 开始训练循环...")
        
        for epoch in range(self.args.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                model_outputs = model(features)
                emotion_logits = model_outputs['emotion_logits']
                speaker_logits = model_outputs.get('speaker_logits', None)
                
                # 计算情感分类损失
                emotion_loss = criterion(emotion_logits, labels)
                
                # 计算对抗损失（如果启用）
                total_loss = emotion_loss
                if speaker_logits is not None and self.args.speaker_adversarial:
                    # 创建假的说话人标签（随机分配）
                    fake_speaker_labels = torch.randint(0, 10, (labels.size(0),), device=labels.device)
                    speaker_loss = criterion(speaker_logits, fake_speaker_labels)
                    total_loss = emotion_loss + self.args.adversarial_weight * speaker_loss
                
                loss = total_loss
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = torch.max(emotion_logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % self.args.log_interval == 0:
                    print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    model_outputs = model(features)
                    emotion_logits = model_outputs['emotion_logits']
                    loss = criterion(emotion_logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(emotion_logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 计算准确率
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            
            # 更新历史
            train_history['loss'].append(train_loss / len(train_loader))
            train_history['acc'].append(train_acc)
            val_history['loss'].append(val_loss / len(val_loader))
            val_history['acc'].append(val_acc)
            
            # 更新学习率
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{self.args.epochs}:')
            print(f'  训练 - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
            print(f'  验证 - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
            print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f'  ✅ 新的最佳验证准确率: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= self.args.patience:
                print(f'  ⏹️  早停触发，验证准确率连续{self.args.patience}轮未提升')
                break
            
            print('-' * 60)
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ 已加载最佳模型 (验证准确率: {best_val_acc:.2f}%)")
        
        # 保存模型
        model_path = os.path.join(self.exp_dir, 'best_enhanced_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': self.args,
            'best_val_acc': best_val_acc,
            'train_history': train_history,
            'val_history': val_history
        }, model_path)
        
        # 保存为pickle格式（兼容GUI）
        model_pkl_path = os.path.join(self.exp_dir, 'best_enhanced_model.pkl')
        with open(model_pkl_path, 'wb') as f:
            pickle.dump({
                'model_state_dict': model.state_dict(),
                'args': self.args,
                'best_val_acc': best_val_acc
            }, f)
        
        print(f"💾 模型已保存: {model_path}")
        print(f"💾 兼容格式已保存: {model_pkl_path}")
        
        # 测试模型
        test_acc, test_f1, test_report = self.evaluate_model(model, test_loader)
        
        # 绘制训练曲线
        self.plot_training_curves(train_history, val_history)
        
        return {
            'model': model,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_report': test_report,
            'train_history': train_history,
            'val_history': val_history
        }
    
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        print("\n📊 评估模型性能...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                model_outputs = model(features)
                emotion_logits = model_outputs['emotion_logits']
                _, predicted = torch.max(emotion_logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        test_acc = accuracy_score(all_labels, all_predictions) * 100
        test_f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        
        # 生成分类报告
        test_report = classification_report(
            all_labels, all_predictions,
            target_names=[self.emotion_labels[i] for i in range(4)],
            digits=4
        )
        
        print(f"🎯 测试准确率: {test_acc:.2f}%")
        print(f"🎯 测试F1分数: {test_f1:.2f}%")
        print("\n📋 详细分类报告:")
        print(test_report)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        return test_acc, test_f1, test_report
    
    def plot_training_curves(self, train_history, val_history):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_history['loss'], label='训练损失', color='blue')
        ax1.plot(val_history['loss'], label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_history['acc'], label='训练准确率', color='blue')
        ax2.plot(val_history['acc'], label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 训练曲线已保存: {os.path.join(self.exp_dir, 'training_curves.png')}")
    
    def plot_confusion_matrix(self, true_labels, pred_labels):
        """绘制混淆矩阵"""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.emotion_labels[i] for i in range(4)],
                   yticklabels=[self.emotion_labels[i] for i in range(4)])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 混淆矩阵已保存: {os.path.join(self.exp_dir, 'confusion_matrix.png')}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='简化的增强IEMOCAP语音情感识别训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./Train_data_org.pickle', help='数据文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    
    # 模型参数
    parser.add_argument('--hidden_layer', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--out_class', type=int, default=4, help='输出类别数')
    parser.add_argument('--dia_layers', type=int, default=2, help='GRU层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--attention', type=bool, default=True, help='是否使用注意力机制')
    parser.add_argument('--speaker_norm', type=bool, default=True, help='是否使用说话人归一化')
    parser.add_argument('--speaker_adversarial', type=bool, default=True, help='是否使用说话人对抗训练')
    parser.add_argument('--freeze_layers', type=int, default=6, help='冻结的HuBERT层数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪最大范数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--T_0', type=int, default=10, help='余弦退火调度器周期')
    
    # 对抗训练参数
    parser.add_argument('--adversarial_weight', type=float, default=0.1, help='对抗损失权重')
    parser.add_argument('--adversarial_warmup', type=int, default=10, help='对抗训练预热轮数')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2正则化权重')
    
    # 其他参数
    parser.add_argument('--cuda', type=bool, default=True, help='是否使用CUDA')
    parser.add_argument('--log_interval', type=int, default=50, help='日志输出间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()

def main():
    """主函数"""
    print("🚀 启动简化的增强IEMOCAP语音情感识别训练系统")
    print("=" * 60)
    
    # 解析参数
    args = parse_args()
    
    # 显示配置
    print("📋 训练配置:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    try:
        # 创建训练器
        trainer = SimpleEnhancedTrainer(args)
        
        # 开始训练
        results = trainer.train_model()
        
        print("\n🎉 训练完成!")
        print(f"✅ 最佳验证准确率: {results['best_val_acc']:.2f}%")
        print(f"✅ 测试准确率: {results['test_acc']:.2f}%")
        print(f"✅ 测试F1分数: {results['test_f1']:.2f}%")
        print(f"📁 结果保存在: {trainer.exp_dir}")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
