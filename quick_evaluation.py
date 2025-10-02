#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速模型评估工具
用于快速测试已训练模型的性能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import argparse
from models import SpeechRecognitionModel
from utils import Get_data
import pickle

# 情感标签映射
EMOTION_LABELS = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
EMOTION_COLORS = ['#e74c3c', '#f39c12', '#95a5a6', '#3498db']

def get_default_args():
    """获取默认参数配置"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention', action='store_true', default=True)
    parser.add_argument('--dia_layers', type=int, default=2)
    parser.add_argument('--hidden_layer', type=int, default=256)
    parser.add_argument('--out_class', type=int, default=4)
    parser.add_argument('--utt_insize', type=int, default=768)
    args, _ = parser.parse_known_args([])
    return args

def load_model_and_data(model_path='model.pkl', data_path='Train_data_org.pickle'):
    """加载模型和数据"""
    print("📥 加载模型和数据...")
    
    # 加载数据
    try:
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        print(f"✅ 数据加载成功: {len(data)} 个样本")
    except FileNotFoundError:
        print(f"❌ 数据文件未找到: {data_path}")
        return None, None
    
    # 加载模型
    args = get_default_args()
    model = SpeechRecognitionModel(args)
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None
    
    return model, data

def evaluate_model_simple(model, data, test_ratio=0.2):
    """简单评估模型性能"""
    print("🔍 开始模型评估...")
    
    # 简单划分数据集
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 获取测试数据加载器
    args = get_default_args()
    _, test_loader, _, _ = Get_data(data, train_indices, test_indices, args)
    
    # 评估模型
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch_idx, (data_batch, target) in enumerate(test_loader):
            data_batch = data_batch.squeeze()
            target = target.squeeze()
            
            # 前向传播
            output, _ = model(data_batch)
            
            # 获取预测
            pred = torch.argmax(output, dim=1)
            probs = torch.softmax(output, dim=1)
            
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(true_labels), np.array(predictions), np.array(probabilities)

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    
    # 每类指标
    f1_per_class = f1_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    precision_per_class = precision_score(y_true, y_pred, average=None)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # UA 和 WA
    wa = accuracy
    ua = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'recall_macro': recall_macro,
        'precision_macro': precision_macro,
        'f1_per_class': f1_per_class,
        'recall_per_class': recall_per_class,
        'precision_per_class': precision_per_class,
        'confusion_matrix': cm,
        'ua': ua,
        'wa': wa
    }

def print_results(metrics):
    """打印评估结果"""
    print("\n" + "="*50)
    print("📊 模型评估结果")
    print("="*50)
    
    print(f"\n🎯 总体性能:")
    print(f"   准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"   F1分数 (Macro):   {metrics['f1_macro']:.4f}")
    print(f"   召回率 (Macro):   {metrics['recall_macro']:.4f}")
    print(f"   精确率 (Macro):   {metrics['precision_macro']:.4f}")
    print(f"   UA (无权重准确率): {metrics['ua']:.4f}")
    print(f"   WA (加权准确率):   {metrics['wa']:.4f}")
    
    print(f"\n🎭 各情感类别性能:")
    for i, emotion in EMOTION_LABELS.items():
        print(f"   {emotion:8s}: F1={metrics['f1_per_class'][i]:.4f}, "
              f"Precision={metrics['precision_per_class'][i]:.4f}, "
              f"Recall={metrics['recall_per_class'][i]:.4f}")

def create_simple_visualization(metrics, save_plot=True):
    """创建简单的可视化图表"""
    print("\n🎨 生成可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 混淆矩阵
    cm = metrics['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
               xticklabels=list(EMOTION_LABELS.values()),
               yticklabels=list(EMOTION_LABELS.values()),
               ax=ax1)
    ax1.set_title('混淆矩阵 (归一化)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')
    
    # 2. 各类别性能对比
    emotions = list(EMOTION_LABELS.values())
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    x = np.arange(len(emotions))
    width = 0.25
    
    ax2.bar(x - width, precision, width, label='精确率', color='#3498db', alpha=0.8)
    ax2.bar(x, recall, width, label='召回率', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width, f1, width, label='F1分数', color='#2ecc71', alpha=0.8)
    
    ax2.set_xlabel('情感类别')
    ax2.set_ylabel('性能指标')
    ax2.set_title('各情感类别性能对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 总体指标对比
    metrics_names = ['准确率', 'F1分数', '召回率', '精确率']
    metrics_values = [metrics['accuracy'], metrics['f1_macro'], 
                     metrics['recall_macro'], metrics['precision_macro']]
    
    bars = ax3.bar(metrics_names, metrics_values, 
                   color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.8)
    ax3.set_title('总体性能指标', fontsize=14, fontweight='bold')
    ax3.set_ylabel('分数')
    ax3.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, value in zip(bars, metrics_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 混淆矩阵原始数值
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
               xticklabels=list(EMOTION_LABELS.values()),
               yticklabels=list(EMOTION_LABELS.values()),
               ax=ax4)
    ax4.set_title('混淆矩阵 (原始数值)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('预测标签')
    ax4.set_ylabel('真实标签')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('quick_evaluation_report.png', dpi=300, bbox_inches='tight')
        print("📊 可视化报告已保存至: quick_evaluation_report.png")
    
    plt.show()

def main():
    """主函数"""
    print("🚀 IEMOCAP 语音情感识别 - 快速评估工具")
    print("=" * 50)
    
    # 加载模型和数据
    model, data = load_model_and_data()
    
    if model is None or data is None:
        print("❌ 加载失败，退出程序")
        return
    
    # 评估模型
    y_true, y_pred, y_probs = evaluate_model_simple(model, data)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 打印结果
    print_results(metrics)
    
    # 创建可视化
    create_simple_visualization(metrics, save_plot=True)
    
    print("\n✅ 快速评估完成！")
    print(f"📊 测试样本数: {len(y_true)}")
    print(f"🎯 主要指标: 准确率={metrics['accuracy']:.4f}, F1分数={metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    main()


