#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨说话人性能评估脚本
对比原始模型和增强模型在跨说话人测试中的性能
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import defaultdict, Counter
import os
import json
from datetime import datetime
import argparse

# 导入模型
from models.enhanced_gru import create_enhanced_model
from models.GRU import SpeechRecognitionModel  # 原始模型
from utils.speaker_independent_data import SpeakerIndependentDataLoader, collate_fn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class SpeakerIndependenceEvaluator:
    """跨说话人独立性评估器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        
        # 情感标签
        self.emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
        self.emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']
        
        # 说话人标签
        self.speaker_labels = {
            0: 'Ses01F', 1: 'Ses01M', 2: 'Ses02F', 3: 'Ses02M', 4: 'Ses03F',
            5: 'Ses03M', 6: 'Ses04F', 7: 'Ses04M', 8: 'Ses05F', 9: 'Ses05M'
        }
        
        # 初始化数据加载器
        self.data_loader = SpeakerIndependentDataLoader(args.data_path)
        
        # 创建结果目录
        self.eval_dir = os.path.join('evaluations', f'speaker_independence_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.eval_dir, exist_ok=True)
        
        print(f"📁 评估结果将保存到: {self.eval_dir}")
    
    def load_model(self, model_path, model_type='enhanced'):
        """加载模型"""
        print(f"📥 加载{model_type}模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if model_type == 'enhanced':
            model = create_enhanced_model(self.args)
        else:
            model = SpeechRecognitionModel(self.args)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ {model_type}模型加载成功")
        return model
    
    def evaluate_model_on_speakers(self, model, test_data, model_name="Model"):
        """评估模型在不同说话人上的性能"""
        print(f"🔍 评估{model_name}在跨说话人测试中的性能...")
        
        # 按说话人分组测试数据
        speaker_data = defaultdict(list)
        for sample in test_data:
            speaker = sample.get('speaker', 'unknown')
            speaker_data[speaker].append(sample)
        
        speaker_results = {}
        all_predictions = []
        all_targets = []
        all_speakers = []
        
        for speaker, samples in speaker_data.items():
            if len(samples) == 0:
                continue
                
            # 创建单个说话人的数据加载器
            speaker_loader = DataLoader(
                samples,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )
            
            # 评估
            speaker_preds = []
            speaker_targets = []
            
            with torch.no_grad():
                for batch in speaker_loader:
                    audio_features = batch['audio_features'].to(self.device)
                    emotion_targets = batch['emotion_labels'].to(self.device)
                    
                    if hasattr(model, 'utterance_net'):
                        # 增强模型
                        outputs = model(audio_features)
                        predictions = torch.argmax(outputs['emotion_logits'], dim=1)
                    else:
                        # 原始模型
                        outputs, _ = model(audio_features)
                        predictions = torch.argmax(outputs, dim=1)
                    
                    speaker_preds.extend(predictions.cpu().numpy())
                    speaker_targets.extend(emotion_targets.cpu().numpy())
            
            # 计算该说话人的性能指标
            accuracy = accuracy_score(speaker_targets, speaker_preds)
            f1 = f1_score(speaker_targets, speaker_preds, average='weighted')
            f1_per_class = f1_score(speaker_targets, speaker_preds, average=None)
            
            speaker_results[speaker] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'f1_per_class': f1_per_class,
                'sample_count': len(speaker_targets),
                'predictions': speaker_preds,
                'targets': speaker_targets
            }
            
            # 收集所有结果
            all_predictions.extend(speaker_preds)
            all_targets.extend(speaker_targets)
            all_speakers.extend([speaker] * len(speaker_preds))
            
            print(f"  {speaker}: 准确率={accuracy:.4f}, F1={f1:.4f}, 样本数={len(speaker_targets)}")
        
        # 计算总体性能
        overall_accuracy = accuracy_score(all_targets, all_predictions)
        overall_f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        print(f"🎯 {model_name}总体性能: 准确率={overall_accuracy:.4f}, F1={overall_f1:.4f}")
        
        return {
            'speaker_results': speaker_results,
            'overall_accuracy': overall_accuracy,
            'overall_f1': overall_f1,
            'all_predictions': all_predictions,
            'all_targets': all_targets,
            'all_speakers': all_speakers
        }
    
    def compare_models(self, enhanced_model_path, original_model_path=None):
        """对比增强模型和原始模型"""
        print("🔄 开始模型对比评估...")
        
        # 创建跨说话人测试集
        _, _, test_data = self.data_loader.create_speaker_independent_splits(0, n_folds=5)
        
        # 评估增强模型
        enhanced_model = self.load_model(enhanced_model_path, 'enhanced')
        enhanced_results = self.evaluate_model_on_speakers(enhanced_model, test_data, "增强模型")
        
        # 评估原始模型（如果提供）
        original_results = None
        if original_model_path and os.path.exists(original_model_path):
            original_model = self.load_model(original_model_path, 'original')
            original_results = self.evaluate_model_on_speakers(original_model, test_data, "原始模型")
        
        # 生成对比报告
        comparison_results = {
            'enhanced': enhanced_results,
            'original': original_results,
            'test_data_info': {
                'total_samples': len(test_data),
                'speaker_distribution': self._get_speaker_distribution(test_data)
            }
        }
        
        # 可视化对比结果
        self.visualize_comparison(comparison_results)
        
        # 保存结果
        self.save_results(comparison_results)
        
        return comparison_results
    
    def _get_speaker_distribution(self, test_data):
        """获取测试数据中的说话人分布"""
        speaker_counts = Counter()
        emotion_by_speaker = defaultdict(Counter)
        
        for sample in test_data:
            speaker = sample.get('speaker', 'unknown')
            emotion = sample.get('emotion', -1)
            
            speaker_counts[speaker] += 1
            emotion_by_speaker[speaker][emotion] += 1
        
        return {
            'speaker_counts': dict(speaker_counts),
            'emotion_by_speaker': {k: dict(v) for k, v in emotion_by_speaker.items()}
        }
    
    def visualize_comparison(self, comparison_results):
        """可视化对比结果"""
        enhanced_results = comparison_results['enhanced']
        original_results = comparison_results['original']
        
        # 1. 说话人性能对比
        self.plot_speaker_performance_comparison(enhanced_results, original_results)
        
        # 2. 混淆矩阵对比
        self.plot_confusion_matrix_comparison(enhanced_results, original_results)
        
        # 3. 情感类别性能对比
        self.plot_emotion_performance_comparison(enhanced_results, original_results)
        
        # 4. 说话人偏见分析
        self.plot_speaker_bias_analysis(enhanced_results, original_results)
    
    def plot_speaker_performance_comparison(self, enhanced_results, original_results):
        """绘制说话人性能对比"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        speakers = list(enhanced_results['speaker_results'].keys())
        enhanced_acc = [enhanced_results['speaker_results'][s]['accuracy'] for s in speakers]
        enhanced_f1 = [enhanced_results['speaker_results'][s]['f1_score'] for s in speakers]
        
        x = np.arange(len(speakers))
        width = 0.35
        
        # 准确率对比
        ax1.bar(x - width/2, enhanced_acc, width, label='增强模型', alpha=0.8, color='skyblue')
        
        if original_results:
            original_acc = [original_results['speaker_results'][s]['accuracy'] for s in speakers]
            ax1.bar(x + width/2, original_acc, width, label='原始模型', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('说话人')
        ax1.set_ylabel('准确率')
        ax1.set_title('各说话人准确率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(speakers, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1分数对比
        ax2.bar(x - width/2, enhanced_f1, width, label='增强模型', alpha=0.8, color='lightgreen')
        
        if original_results:
            original_f1 = [original_results['speaker_results'][s]['f1_score'] for s in speakers]
            ax2.bar(x + width/2, original_f1, width, label='原始模型', alpha=0.8, color='orange')
        
        ax2.set_xlabel('说话人')
        ax2.set_ylabel('F1分数')
        ax2.set_title('各说话人F1分数对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(speakers, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, 'speaker_performance_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix_comparison(self, enhanced_results, original_results):
        """绘制混淆矩阵对比"""
        if original_results is None:
            # 只有增强模型的混淆矩阵
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            cm = confusion_matrix(enhanced_results['all_targets'], enhanced_results['all_predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.emotion_labels.values(),
                       yticklabels=self.emotion_labels.values())
            ax.set_title(f'增强模型混淆矩阵\\n准确率: {enhanced_results["overall_accuracy"]:.4f}')
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
        else:
            # 对比两个模型的混淆矩阵
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 增强模型
            cm1 = confusion_matrix(enhanced_results['all_targets'], enhanced_results['all_predictions'])
            sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=self.emotion_labels.values(),
                       yticklabels=self.emotion_labels.values())
            ax1.set_title(f'增强模型\\n准确率: {enhanced_results["overall_accuracy"]:.4f}')
            ax1.set_xlabel('预测标签')
            ax1.set_ylabel('真实标签')
            
            # 原始模型
            cm2 = confusion_matrix(original_results['all_targets'], original_results['all_predictions'])
            sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds', ax=ax2,
                       xticklabels=self.emotion_labels.values(),
                       yticklabels=self.emotion_labels.values())
            ax2.set_title(f'原始模型\\n准确率: {original_results["overall_accuracy"]:.4f}')
            ax2.set_xlabel('预测标签')
            ax2.set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, 'confusion_matrix_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_emotion_performance_comparison(self, enhanced_results, original_results):
        """绘制情感类别性能对比"""
        # 计算各情感类别的平均性能
        enhanced_emotion_f1 = self._compute_emotion_f1_by_speaker(enhanced_results)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        emotion_names = list(self.emotion_labels.values())
        x = np.arange(len(emotion_names))
        width = 0.35
        
        # 增强模型
        enhanced_means = [np.mean(enhanced_emotion_f1[i]) for i in range(len(emotion_names))]
        enhanced_stds = [np.std(enhanced_emotion_f1[i]) for i in range(len(emotion_names))]
        
        bars1 = ax.bar(x - width/2, enhanced_means, width, yerr=enhanced_stds,
                      label='增强模型', alpha=0.8, color=self.emotion_colors, capsize=5)
        
        if original_results:
            original_emotion_f1 = self._compute_emotion_f1_by_speaker(original_results)
            original_means = [np.mean(original_emotion_f1[i]) for i in range(len(emotion_names))]
            original_stds = [np.std(original_emotion_f1[i]) for i in range(len(emotion_names))]
            
            bars2 = ax.bar(x + width/2, original_means, width, yerr=original_stds,
                          label='原始模型', alpha=0.6, color=self.emotion_colors, capsize=5)
        
        ax.set_xlabel('情感类别')
        ax.set_ylabel('F1分数')
        ax.set_title('各情感类别跨说话人性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(emotion_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, mean, std in zip(bars1, enhanced_means, enhanced_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, 'emotion_performance_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_speaker_bias_analysis(self, enhanced_results, original_results):
        """绘制说话人偏见分析"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 说话人性能方差分析
        speakers = list(enhanced_results['speaker_results'].keys())
        enhanced_accs = [enhanced_results['speaker_results'][s]['accuracy'] for s in speakers]
        enhanced_f1s = [enhanced_results['speaker_results'][s]['f1_score'] for s in speakers]
        
        enhanced_acc_var = np.var(enhanced_accs)
        enhanced_f1_var = np.var(enhanced_f1s)
        
        metrics = ['准确率方差', 'F1分数方差']
        enhanced_vars = [enhanced_acc_var, enhanced_f1_var]
        
        if original_results:
            original_accs = [original_results['speaker_results'][s]['accuracy'] for s in speakers]
            original_f1s = [original_results['speaker_results'][s]['f1_score'] for s in speakers]
            original_acc_var = np.var(original_accs)
            original_f1_var = np.var(original_f1s)
            original_vars = [original_acc_var, original_f1_var]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, enhanced_vars, width, label='增强模型', alpha=0.8, color='skyblue')
            ax1.bar(x + width/2, original_vars, width, label='原始模型', alpha=0.8, color='lightcoral')
            ax1.set_xlabel('指标')
            ax1.set_ylabel('方差')
            ax1.set_title('说话人间性能方差对比\\n(方差越小说明跨说话人泛化越好)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.bar(metrics, enhanced_vars, alpha=0.8, color=['skyblue', 'lightgreen'])
            ax1.set_xlabel('指标')
            ax1.set_ylabel('方差')
            ax1.set_title('增强模型说话人间性能方差')
            ax1.grid(True, alpha=0.3)
        
        # 2. 性能分布箱线图
        if original_results:
            ax2.boxplot([enhanced_accs, original_accs], labels=['增强模型', '原始模型'])
            ax2.set_ylabel('准确率')
            ax2.set_title('说话人准确率分布对比')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.boxplot([enhanced_accs], labels=['增强模型'])
            ax2.set_ylabel('准确率')
            ax2.set_title('增强模型说话人准确率分布')
            ax2.grid(True, alpha=0.3)
        
        # 3. 性别差异分析
        male_speakers = [s for s in speakers if 'M' in s]
        female_speakers = [s for s in speakers if 'F' in s]
        
        enhanced_male_acc = np.mean([enhanced_results['speaker_results'][s]['accuracy'] for s in male_speakers])
        enhanced_female_acc = np.mean([enhanced_results['speaker_results'][s]['accuracy'] for s in female_speakers])
        
        gender_labels = ['男性说话人', '女性说话人']
        enhanced_gender_acc = [enhanced_male_acc, enhanced_female_acc]
        
        if original_results:
            original_male_acc = np.mean([original_results['speaker_results'][s]['accuracy'] for s in male_speakers])
            original_female_acc = np.mean([original_results['speaker_results'][s]['accuracy'] for s in female_speakers])
            original_gender_acc = [original_male_acc, original_female_acc]
            
            x = np.arange(len(gender_labels))
            width = 0.35
            
            ax3.bar(x - width/2, enhanced_gender_acc, width, label='增强模型', alpha=0.8, color='skyblue')
            ax3.bar(x + width/2, original_gender_acc, width, label='原始模型', alpha=0.8, color='lightcoral')
        else:
            ax3.bar(gender_labels, enhanced_gender_acc, alpha=0.8, color=['lightblue', 'pink'])
        
        ax3.set_xlabel('说话人性别')
        ax3.set_ylabel('平均准确率')
        ax3.set_title('性别差异分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 会话差异分析
        session_acc = defaultdict(list)
        for speaker in speakers:
            session = speaker[:7]  # 提取会话信息 (e.g., 'Ses01F' -> 'Session')
            session_acc[session].append(enhanced_results['speaker_results'][speaker]['accuracy'])
        
        sessions = list(session_acc.keys())
        session_means = [np.mean(session_acc[s]) for s in sessions]
        
        ax4.bar(sessions, session_means, alpha=0.8, color='lightgreen')
        ax4.set_xlabel('会话')
        ax4.set_ylabel('平均准确率')
        ax4.set_title('会话间性能差异')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, 'speaker_bias_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _compute_emotion_f1_by_speaker(self, results):
        """计算各说话人在每个情感类别上的F1分数"""
        emotion_f1_by_speaker = [[] for _ in range(4)]  # 4个情感类别
        
        for speaker, speaker_result in results['speaker_results'].items():
            f1_per_class = speaker_result['f1_per_class']
            for i, f1 in enumerate(f1_per_class):
                emotion_f1_by_speaker[i].append(f1)
        
        return emotion_f1_by_speaker
    
    def save_results(self, comparison_results):
        """保存评估结果"""
        # 保存详细结果
        results_path = os.path.join(self.eval_dir, 'detailed_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成简化的对比报告
        enhanced_results = comparison_results['enhanced']
        original_results = comparison_results['original']
        
        report = {
            'evaluation_time': datetime.now().isoformat(),
            'test_data_info': comparison_results['test_data_info'],
            'enhanced_model': {
                'overall_accuracy': enhanced_results['overall_accuracy'],
                'overall_f1': enhanced_results['overall_f1'],
                'speaker_performance_variance': {
                    'accuracy_var': np.var([r['accuracy'] for r in enhanced_results['speaker_results'].values()]),
                    'f1_var': np.var([r['f1_score'] for r in enhanced_results['speaker_results'].values()])
                }
            }
        }
        
        if original_results:
            report['original_model'] = {
                'overall_accuracy': original_results['overall_accuracy'],
                'overall_f1': original_results['overall_f1'],
                'speaker_performance_variance': {
                    'accuracy_var': np.var([r['accuracy'] for r in original_results['speaker_results'].values()]),
                    'f1_var': np.var([r['f1_score'] for r in original_results['speaker_results'].values()])
                }
            }
            
            # 计算改进幅度
            report['improvement'] = {
                'accuracy_improvement': enhanced_results['overall_accuracy'] - original_results['overall_accuracy'],
                'f1_improvement': enhanced_results['overall_f1'] - original_results['overall_f1'],
                'variance_reduction': {
                    'accuracy': report['original_model']['speaker_performance_variance']['accuracy_var'] - 
                               report['enhanced_model']['speaker_performance_variance']['accuracy_var'],
                    'f1': report['original_model']['speaker_performance_variance']['f1_var'] - 
                          report['enhanced_model']['speaker_performance_variance']['f1_var']
                }
            }
        
        # 保存简化报告
        report_path = os.path.join(self.eval_dir, 'comparison_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print("\\n" + "="*60)
        print("📊 跨说话人性能评估总结")
        print("="*60)
        
        print(f"增强模型性能:")
        print(f"  总体准确率: {enhanced_results['overall_accuracy']:.4f}")
        print(f"  总体F1分数: {enhanced_results['overall_f1']:.4f}")
        print(f"  说话人间准确率方差: {report['enhanced_model']['speaker_performance_variance']['accuracy_var']:.6f}")
        
        if original_results:
            print(f"\\n原始模型性能:")
            print(f"  总体准确率: {original_results['overall_accuracy']:.4f}")
            print(f"  总体F1分数: {original_results['overall_f1']:.4f}")
            print(f"  说话人间准确率方差: {report['original_model']['speaker_performance_variance']['accuracy_var']:.6f}")
            
            print(f"\\n改进效果:")
            print(f"  准确率提升: {report['improvement']['accuracy_improvement']:.4f}")
            print(f"  F1分数提升: {report['improvement']['f1_improvement']:.4f}")
            print(f"  方差减少: {report['improvement']['variance_reduction']['accuracy']:.6f}")
        
        print(f"\\n📁 详细结果保存在: {self.eval_dir}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="跨说话人性能评估")
    
    parser.add_argument('--enhanced_model_path', type=str, required=True,
                       help='增强模型路径')
    parser.add_argument('--original_model_path', type=str, default=None,
                       help='原始模型路径（可选）')
    parser.add_argument('--data_path', type=str, default='./Train_data_org.pickle',
                       help='数据文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='是否使用CUDA')
    
    # 模型参数（需要与训练时一致）
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
    
    return parser.parse_args()

def main():
    """主函数"""
    print("🔍 启动跨说话人性能评估系统")
    print("="*60)
    
    args = parse_args()
    
    # 创建评估器
    evaluator = SpeakerIndependenceEvaluator(args)
    
    # 运行评估
    try:
        results = evaluator.compare_models(
            enhanced_model_path=args.enhanced_model_path,
            original_model_path=args.original_model_path
        )
        
        print("\\n🎉 评估完成!")
        
    except Exception as e:
        print(f"\\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


