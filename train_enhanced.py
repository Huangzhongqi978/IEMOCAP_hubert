#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的IEMOCAP语音情感识别训练代码
专注于跨说话人泛化和高准确率优化

主要改进:
1. 说话人无关训练策略
2. 增强的GRU架构 + 说话人归一化
3. 对抗训练减少说话人偏见
4. 先进的训练策略和数据增强
5. 全面的可视化和评估
"""

import argparse
import pickle
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300

# 导入自定义模块
from models.enhanced_gru import create_enhanced_model
from utils.speaker_independent_data import SpeakerIndependentDataLoader, collate_fn

warnings.filterwarnings('ignore')

# 解决中文编码问题
import locale
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese_China.UTF-8')
    except:
        print("⚠️ 无法设置中文本地化，可能影响中文显示")

class AdvancedTrainer:
    """高级训练器 - 包含所有优化策略"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        
        # 情感标签
        self.emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
        self.emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']
        
        # 初始化数据加载器
        self.data_loader = SpeakerIndependentDataLoader(args.data_path)
        
        # 创建实验目录
        self.exp_name = f"enhanced_emotion_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join('experiments', self.exp_name)
        self.plots_dir = os.path.join(self.exp_dir, 'plots')
        self.models_dir = os.path.join(self.exp_dir, 'models')
        
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 保存配置
        self.save_config()
        
        # 训练历史记录
        self.training_history = {
            'fold_results': [],
            'best_metrics': {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'fold': -1
            }
        }
        
        print(f"🚀 增强训练器初始化完成")
        print(f"📁 实验目录: {self.exp_dir}")
        print(f"🖥️  设备: {self.device}")
    
    def save_config(self):
        """保存实验配置"""
        config = {
            'args': vars(self.args),
            'exp_name': self.exp_name,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def create_model(self):
        """创建增强模型"""
        model = create_enhanced_model(self.args)
        model = model.to(self.device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        return model
    
    def create_optimizers_and_schedulers(self, model):
        """创建优化器和学习率调度器"""
        # 分组参数 - 不同组使用不同学习率
        hubert_params = []
        gru_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'feature_extractor' in name:
                hubert_params.append(param)
            elif 'utterance_net' in name:
                gru_params.append(param)
            else:
                classifier_params.append(param)
        
        # 创建参数组
        param_groups = [
            {'params': hubert_params, 'lr': self.args.lr * 0.1, 'name': 'hubert'},  # HuBERT用更小学习率
            {'params': gru_params, 'lr': self.args.lr, 'name': 'gru'},
            {'params': classifier_params, 'lr': self.args.lr * 2, 'name': 'classifier'}  # 分类器用更大学习率
        ]
        
        # 主优化器
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.args.T_0,
            T_mult=2,
            eta_min=1e-7
        )
        
        # 备用调度器 - 基于验证损失的自适应调度
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        
        return optimizer, scheduler, plateau_scheduler
    
    def compute_loss(self, model_outputs, targets, speaker_targets, alpha=1.0):
        """计算综合损失"""
        emotion_logits = model_outputs['emotion_logits']
        speaker_logits = model_outputs['speaker_logits']
        
        # 情感分类损失（主要任务）
        emotion_loss = F.cross_entropy(emotion_logits, targets)
        
        total_loss = emotion_loss
        loss_dict = {'emotion_loss': emotion_loss.item()}
        
        # 说话人对抗损失
        if speaker_logits is not None and self.args.speaker_adversarial:
            speaker_loss = F.cross_entropy(speaker_logits, speaker_targets)
            total_loss += self.args.adversarial_weight * speaker_loss
            loss_dict['speaker_loss'] = speaker_loss.item()
        
        # 正则化损失
        if self.args.l2_reg > 0:
            l2_loss = sum(torch.norm(p, 2) for p in model_outputs.get('regularization_params', []))
            total_loss += self.args.l2_reg * l2_loss
            loss_dict['l2_loss'] = l2_loss.item() if isinstance(l2_loss, torch.Tensor) else l2_loss
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, epoch, alpha):
        """训练一个epoch"""
        model.train()
        
        total_loss = 0.0
        total_emotion_loss = 0.0
        total_speaker_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据移到设备
            audio_features = batch['audio_features'].to(self.device)
            emotion_targets = batch['emotion_labels'].to(self.device)
            speaker_targets = batch['speaker_labels'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(audio_features, alpha=alpha)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(outputs, emotion_targets, speaker_targets, alpha)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            
            optimizer.step()
            
            # 统计
            total_loss += loss_dict['total_loss']
            total_emotion_loss += loss_dict['emotion_loss']
            if 'speaker_loss' in loss_dict:
                total_speaker_loss += loss_dict['speaker_loss']
            
            # 计算准确率
            predictions = torch.argmax(outputs['emotion_logits'], dim=1)
            correct_predictions += (predictions == emotion_targets).sum().item()
            total_samples += emotion_targets.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(emotion_targets.cpu().numpy())
            
            # 打印进度
            if batch_idx % self.args.log_interval == 0:
                print(f'训练 Epoch: {epoch} [{batch_idx * len(audio_features)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\\t'
                      f'损失: {loss.item():.6f}')
        
        # 计算epoch统计
        avg_loss = total_loss / len(train_loader)
        avg_emotion_loss = total_emotion_loss / len(train_loader)
        avg_speaker_loss = total_speaker_loss / len(train_loader) if total_speaker_loss > 0 else 0
        accuracy = correct_predictions / total_samples
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # 更新学习率
        scheduler.step()
        
        return {
            'loss': avg_loss,
            'emotion_loss': avg_emotion_loss,
            'speaker_loss': avg_speaker_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, model, data_loader, alpha=0.0):
        """评估模型"""
        model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        all_speaker_predictions = []
        all_speaker_targets = []
        all_features = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 数据移到设备
                audio_features = batch['audio_features'].to(self.device)
                emotion_targets = batch['emotion_labels'].to(self.device)
                speaker_targets = batch['speaker_labels'].to(self.device)
                
                # 前向传播
                outputs = model(audio_features, alpha=alpha)
                
                # 计算损失
                loss, _ = self.compute_loss(outputs, emotion_targets, speaker_targets, alpha)
                total_loss += loss.item()
                
                # 预测
                emotion_predictions = torch.argmax(outputs['emotion_logits'], dim=1)
                correct_predictions += (emotion_predictions == emotion_targets).sum().item()
                total_samples += emotion_targets.size(0)
                
                # 收集结果
                all_predictions.extend(emotion_predictions.cpu().numpy())
                all_targets.extend(emotion_targets.cpu().numpy())
                all_features.append(outputs['global_features'].cpu().numpy())
                
                if outputs['speaker_logits'] is not None:
                    speaker_predictions = torch.argmax(outputs['speaker_logits'], dim=1)
                    all_speaker_predictions.extend(speaker_predictions.cpu().numpy())
                    all_speaker_targets.extend(speaker_targets.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        
        # 各类别F1分数
        f1_per_class = f1_score(all_targets, all_predictions, average=None)
        
        # 说话人分类准确率（如果有）
        speaker_accuracy = 0.0
        if all_speaker_predictions:
            speaker_accuracy = accuracy_score(all_speaker_targets, all_speaker_predictions)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class,
            'speaker_accuracy': speaker_accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'features': np.vstack(all_features) if all_features else None
        }
        
        return results
    
    def train_fold(self, fold_idx):
        """训练单个fold"""
        print(f"\\n{'='*60}")
        print(f"🔄 开始训练第 {fold_idx+1}/{self.args.n_folds} 折")
        print(f"{'='*60}")
        
        # 创建数据划分
        train_data, val_data, test_data = self.data_loader.create_speaker_independent_splits(
            fold_idx, self.args.n_folds
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_data, 
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # 创建模型
        model = self.create_model()
        
        # 创建优化器和调度器
        optimizer, scheduler, plateau_scheduler = self.create_optimizers_and_schedulers(model)
        
        # 训练历史
        fold_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }
        
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        
        # 训练循环
        for epoch in range(1, self.args.epochs + 1):
            # 计算对抗训练强度
            alpha = min(1.0, epoch / self.args.adversarial_warmup) if self.args.speaker_adversarial else 0.0
            
            # 训练
            train_results = self.train_epoch(model, train_loader, optimizer, scheduler, epoch, alpha)
            
            # 验证
            val_results = self.evaluate(model, val_loader, alpha=0.0)
            
            # 记录历史
            fold_history['train_loss'].append(train_results['loss'])
            fold_history['train_acc'].append(train_results['accuracy'])
            fold_history['train_f1'].append(train_results['f1_score'])
            fold_history['val_loss'].append(val_results['loss'])
            fold_history['val_acc'].append(val_results['accuracy'])
            fold_history['val_f1'].append(val_results['f1_score'])
            fold_history['learning_rates'].append(train_results['learning_rate'])
            
            # 更新plateau调度器
            plateau_scheduler.step(val_results['loss'])
            
            # 保存最佳模型
            if val_results['f1_score'] > best_val_f1:
                best_val_f1 = val_results['f1_score']
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
                print(f"✅ 新的最佳验证F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= self.args.patience:
                print(f"⏹️ 早停触发 (patience={self.args.patience})")
                break
            
            # 打印epoch结果
            print(f"Epoch {epoch:3d} | "
                  f"训练: Loss={train_results['loss']:.4f}, Acc={train_results['accuracy']:.4f}, F1={train_results['f1_score']:.4f} | "
                  f"验证: Loss={val_results['loss']:.4f}, Acc={val_results['accuracy']:.4f}, F1={val_results['f1_score']:.4f} | "
                  f"LR={train_results['learning_rate']:.2e}")
        
        # 加载最佳模型进行测试
        model.load_state_dict(best_model_state)
        
        # 最终测试
        test_results = self.evaluate(model, test_loader, alpha=0.0)
        
        print(f"\\n🎯 第 {fold_idx+1} 折最终结果:")
        print(f"   测试准确率: {test_results['accuracy']:.4f}")
        print(f"   测试F1分数: {test_results['f1_score']:.4f}")
        print(f"   测试精确率: {test_results['precision']:.4f}")
        print(f"   测试召回率: {test_results['recall']:.4f}")
        
        # 保存模型
        model_path = os.path.join(self.models_dir, f'best_model_fold_{fold_idx}.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'fold_idx': fold_idx,
            'test_results': test_results,
            'args': self.args
        }, model_path)
        
        # 保存GUI兼容的.pkl格式模型
        pkl_model_path = os.path.join(self.models_dir, f'model_fold_{fold_idx}.pkl')
        model.load_state_dict(best_model_state)
        model.eval()
        torch.save(model, pkl_model_path)
        print(f"✅ 已保存GUI兼容模型: {pkl_model_path}")
        
        # 生成可视化
        self.plot_fold_results(fold_idx, fold_history, test_results, test_loader, model)
        
        return {
            'fold_idx': fold_idx,
            'test_results': test_results,
            'fold_history': fold_history,
            'model_path': model_path
        }
    
    def plot_fold_results(self, fold_idx, fold_history, test_results, test_loader, model):
        """绘制单个fold的结果"""
        # 1. 训练曲线
        self.plot_training_curves(fold_idx, fold_history)
        
        # 2. 混淆矩阵
        self.plot_confusion_matrix(fold_idx, test_results)
        
        # 3. 特征可视化
        if test_results['features'] is not None:
            self.plot_feature_visualization(fold_idx, test_results)
        
        # 4. 注意力可视化（如果有）
        if self.args.attention:
            self.plot_attention_visualization(fold_idx, test_loader, model)
    
    def plot_training_curves(self, fold_idx, fold_history):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(fold_history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, fold_history['train_loss'], 'b-', label='训练损失', alpha=0.8)
        axes[0, 0].plot(epochs, fold_history['val_loss'], 'r-', label='验证损失', alpha=0.8)
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(epochs, fold_history['train_acc'], 'b-', label='训练准确率', alpha=0.8)
        axes[0, 1].plot(epochs, fold_history['val_acc'], 'r-', label='验证准确率', alpha=0.8)
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1分数曲线
        axes[1, 0].plot(epochs, fold_history['train_f1'], 'b-', label='训练F1', alpha=0.8)
        axes[1, 0].plot(epochs, fold_history['val_f1'], 'r-', label='验证F1', alpha=0.8)
        axes[1, 0].set_title('F1分数曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1, 1].plot(epochs, fold_history['learning_rates'], 'g-', alpha=0.8)
        axes[1, 1].set_title('学习率曲线')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'training_curves_fold_{fold_idx}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, fold_idx, test_results):
        """绘制混淆矩阵"""
        cm = confusion_matrix(test_results['targets'], test_results['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels.values(),
                   yticklabels=self.emotion_labels.values())
        plt.title(f'第 {fold_idx+1} 折混淆矩阵\\n准确率: {test_results["accuracy"]:.4f}, F1: {test_results["f1_score"]:.4f}')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'confusion_matrix_fold_{fold_idx}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_visualization(self, fold_idx, test_results):
        """绘制特征可视化"""
        features = test_results['features']
        targets = test_results['targets']
        
        # t-SNE降维
        print(f"🔍 对第 {fold_idx+1} 折特征进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # 绘制
        plt.figure(figsize=(12, 8))
        for emotion_id, emotion_name in self.emotion_labels.items():
            mask = np.array(targets) == emotion_id
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=self.emotion_colors[emotion_id], label=emotion_name, 
                       alpha=0.7, s=50)
        
        plt.title(f'第 {fold_idx+1} 折特征分布 (t-SNE)')
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'feature_visualization_fold_{fold_idx}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_visualization(self, fold_idx, test_loader, model):
        """绘制注意力可视化"""
        model.eval()
        
        # 获取几个样本的注意力权重
        with torch.no_grad():
            batch = next(iter(test_loader))
            audio_features = batch['audio_features'][:4].to(self.device)  # 取前4个样本
            targets = batch['emotion_labels'][:4]
            
            outputs = model(audio_features)
            attention_weights = outputs.get('attention_weights')
            
            if attention_weights is not None:
                attention_weights = attention_weights.cpu().numpy()
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i in range(min(4, len(attention_weights))):
                    # 取第一个头的注意力权重
                    attn = attention_weights[i][0]  # [seq_len, seq_len]
                    
                    im = axes[i].imshow(attn, cmap='YlOrRd', aspect='auto')
                    axes[i].set_title(f'样本 {i+1} - 真实: {self.emotion_labels[targets[i].item()]}')
                    axes[i].set_xlabel('时间步')
                    axes[i].set_ylabel('时间步')
                    plt.colorbar(im, ax=axes[i])
                
                plt.suptitle(f'第 {fold_idx+1} 折注意力权重可视化')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'attention_weights_fold_{fold_idx}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def run_cross_validation(self):
        """运行完整的交叉验证"""
        print("🚀 开始跨说话人交叉验证训练...")
        
        all_fold_results = []
        
        for fold_idx in range(self.args.n_folds):
            fold_result = self.train_fold(fold_idx)
            all_fold_results.append(fold_result)
            
            # 更新最佳结果
            test_f1 = fold_result['test_results']['f1_score']
            if test_f1 > self.training_history['best_metrics']['f1_score']:
                self.training_history['best_metrics'].update({
                    'accuracy': fold_result['test_results']['accuracy'],
                    'f1_score': test_f1,
                    'fold': fold_idx
                })
        
        # 保存所有结果
        self.training_history['fold_results'] = all_fold_results
        
        # 计算总体统计
        self.compute_overall_statistics()
        
        # 生成综合报告
        self.generate_comprehensive_report()
        
        print("🎉 训练完成!")
        return self.training_history
    
    def compute_overall_statistics(self):
        """计算总体统计"""
        fold_results = self.training_history['fold_results']
        
        # 提取各项指标
        accuracies = [r['test_results']['accuracy'] for r in fold_results]
        f1_scores = [r['test_results']['f1_score'] for r in fold_results]
        precisions = [r['test_results']['precision'] for r in fold_results]
        recalls = [r['test_results']['recall'] for r in fold_results]
        
        # 计算统计量
        stats = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'f1_score': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores)
            },
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'min': np.min(precisions),
                'max': np.max(precisions)
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'min': np.min(recalls),
                'max': np.max(recalls)
            }
        }
        
        self.training_history['overall_stats'] = stats
        
        # 打印结果
        print("\\n" + "="*60)
        print("📊 跨说话人交叉验证总体结果")
        print("="*60)
        
        for metric, values in stats.items():
            print(f"{metric.upper():>12}: {values['mean']:.4f} ± {values['std']:.4f} "
                  f"(min: {values['min']:.4f}, max: {values['max']:.4f})")
        
        print(f"\\n🏆 最佳模型: 第 {self.training_history['best_metrics']['fold']+1} 折")
        print(f"   最佳准确率: {self.training_history['best_metrics']['accuracy']:.4f}")
        print(f"   最佳F1分数: {self.training_history['best_metrics']['f1_score']:.4f}")
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        fold_results = self.training_history['fold_results']
        stats = self.training_history['overall_stats']
        
        # 创建综合图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 各fold性能对比
        ax1 = plt.subplot(3, 3, 1)
        fold_indices = range(1, len(fold_results) + 1)
        accuracies = [r['test_results']['accuracy'] for r in fold_results]
        f1_scores = [r['test_results']['f1_score'] for r in fold_results]
        
        x = np.arange(len(fold_indices))
        width = 0.35
        
        ax1.bar(x - width/2, accuracies, width, label='准确率', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, f1_scores, width, label='F1分数', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('性能指标')
        ax1.set_title('各Fold性能对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Fold {i}' for i in fold_indices])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 性能指标箱线图
        ax2 = plt.subplot(3, 3, 2)
        metrics_data = [
            [r['test_results']['accuracy'] for r in fold_results],
            [r['test_results']['f1_score'] for r in fold_results],
            [r['test_results']['precision'] for r in fold_results],
            [r['test_results']['recall'] for r in fold_results]
        ]
        
        ax2.boxplot(metrics_data, labels=['准确率', 'F1分数', '精确率', '召回率'])
        ax2.set_title('性能指标分布')
        ax2.set_ylabel('数值')
        ax2.grid(True, alpha=0.3)
        
        # 3. 各情感类别平均性能
        ax3 = plt.subplot(3, 3, 3)
        # 计算各类别平均F1分数
        all_f1_per_class = []
        for fold_result in fold_results:
            all_f1_per_class.append(fold_result['test_results']['f1_per_class'])
        
        mean_f1_per_class = np.mean(all_f1_per_class, axis=0)
        std_f1_per_class = np.std(all_f1_per_class, axis=0)
        
        emotion_names = list(self.emotion_labels.values())
        x_pos = np.arange(len(emotion_names))
        
        bars = ax3.bar(x_pos, mean_f1_per_class, yerr=std_f1_per_class, 
                      capsize=5, alpha=0.8, color=self.emotion_colors)
        ax3.set_xlabel('情感类别')
        ax3.set_ylabel('平均F1分数')
        ax3.set_title('各情感类别性能')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(emotion_names)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value, std in zip(bars, mean_f1_per_class, std_f1_per_class):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4-9. 各fold训练曲线（小图）
        for i, fold_result in enumerate(fold_results[:6]):  # 最多显示6个fold
            ax = plt.subplot(3, 3, i + 4)
            fold_history = fold_result['fold_history']
            epochs = range(1, len(fold_history['train_loss']) + 1)
            
            ax.plot(epochs, fold_history['val_acc'], 'b-', alpha=0.7, label='验证准确率')
            ax.plot(epochs, fold_history['val_f1'], 'r-', alpha=0.7, label='验证F1')
            ax.set_title(f'Fold {i+1} 训练曲线')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('性能')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('跨说话人语音情感识别 - 综合性能分析报告', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'comprehensive_analysis_report.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细报告
        report_path = os.path.join(self.exp_dir, 'detailed_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 详细报告已保存: {report_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="增强的IEMOCAP语音情感识别训练")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./Train_data_org.pickle',
                       help='数据文件路径')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    
    # 模型参数
    parser.add_argument('--hidden_layer', type=int, default=128,
                       help='隐藏层大小')
    parser.add_argument('--out_class', type=int, default=4,
                       help='输出类别数')
    parser.add_argument('--dia_layers', type=int, default=2,
                       help='GRU层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比例')
    parser.add_argument('--attention', action='store_true', default=True,
                       help='是否使用注意力机制')
    parser.add_argument('--speaker_norm', action='store_true', default=True,
                       help='是否使用说话人归一化')
    parser.add_argument('--speaker_adversarial', action='store_true', default=True,
                       help='是否使用说话人对抗训练')
    parser.add_argument('--freeze_layers', type=int, default=6,
                       help='冻结HuBERT的层数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停patience')
    parser.add_argument('--T_0', type=int, default=10,
                       help='余弦退火重启周期')
    
    # 对抗训练参数
    parser.add_argument('--adversarial_weight', type=float, default=0.1,
                       help='对抗损失权重')
    parser.add_argument('--adversarial_warmup', type=int, default=10,
                       help='对抗训练预热轮数')
    parser.add_argument('--l2_reg', type=float, default=1e-5,
                       help='L2正则化强度')
    
    # 交叉验证参数
    parser.add_argument('--n_folds', type=int, default=1,
                       help='交叉验证折数')
    
    # 其他参数
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='是否使用CUDA')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='日志打印间隔')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """主函数"""
    print("🚀 启动增强的IEMOCAP语音情感识别训练系统")
    print("="*60)
    
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print("📋 训练配置:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    print("="*60)
    
    # 创建训练器
    trainer = AdvancedTrainer(args)
    
    # 运行训练
    try:
        results = trainer.run_cross_validation()
        
        print("\\n🎉 训练成功完成!")
        print(f"📁 实验结果保存在: {trainer.exp_dir}")
        print(f"🏆 最佳性能: 准确率 {results['best_metrics']['accuracy']:.4f}, "
              f"F1分数 {results['best_metrics']['f1_score']:.4f}")
        
        return results
        
    except KeyboardInterrupt:
        print("\\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
