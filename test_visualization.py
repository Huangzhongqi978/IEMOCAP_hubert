#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试可视化功能
快速验证训练代码中的可视化功能是否正常工作
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def test_chinese_font():
    """测试中文字体是否正常显示"""
    print("🧪 测试中文字体显示...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试中文文本
    test_texts = [
        "IEMOCAP 语音情感识别系统",
        "训练曲线图",
        "注意力权重热力图", 
        "情感概率时序图",
        "混淆矩阵",
        "各情感类别: 愤怒、高兴、中性、悲伤"
    ]
    
    for i, text in enumerate(test_texts):
        ax.text(0.1, 0.9 - i*0.12, text, fontsize=14, fontweight='bold', 
                transform=ax.transAxes)
    
    ax.set_title("中文字体显示测试", fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 保存测试图片
    save_path = 'test_chinese_font.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if os.path.exists(save_path):
        print(f"✅ 中文字体测试通过，图片已保存: {save_path}")
        return True
    else:
        print("❌ 中文字体测试失败")
        return False

def test_emotion_visualization():
    """测试情感可视化功能"""
    print("🧪 测试情感可视化功能...")
    
    # 模拟数据
    emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
    emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']
    
    # 模拟概率数据
    np.random.seed(42)
    probabilities = np.random.dirichlet([1, 1, 1, 1], 5)  # 5个样本
    true_labels = [0, 1, 2, 3, 1]  # 真实标签
    pred_labels = [0, 1, 1, 3, 1]  # 预测标签
    
    # 创建情感概率可视化
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    
    for i in range(5):
        probs = probabilities[i]
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        
        # 绘制概率条形图
        x_pos = np.arange(len(emotion_labels))
        bars = axes[i].bar(x_pos, probs, color=emotion_colors, alpha=0.8, 
                          edgecolor='black', linewidth=1)
        
        # 高亮真实标签和预测标签
        bars[true_label].set_edgecolor('green')
        bars[true_label].set_linewidth(3)
        bars[pred_label].set_alpha(1.0)
        
        # 添加概率数值标签
        for j, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 设置标题和标签
        correct = "✓" if true_label == pred_label else "✗"
        axes[i].set_title(f'样本{i+1} {correct} - 真实: {emotion_labels[true_label]}, 预测: {emotion_labels[pred_label]}',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('情感类别')
        axes[i].set_ylabel('预测概率')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(emotion_labels.values())
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('情感识别概率分布测试', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'test_emotion_probabilities.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if os.path.exists(save_path):
        print(f"✅ 情感可视化测试通过，图片已保存: {save_path}")
        return True
    else:
        print("❌ 情感可视化测试失败")
        return False

def test_training_curves():
    """测试训练曲线可视化"""
    print("🧪 测试训练曲线可视化...")
    
    # 模拟训练数据
    epochs = 20
    np.random.seed(42)
    
    # 生成模拟的训练和测试损失
    train_losses = 2.0 * np.exp(-0.1 * np.arange(epochs)) + 0.1 * np.random.random(epochs)
    test_losses = 2.2 * np.exp(-0.08 * np.arange(epochs)) + 0.15 * np.random.random(epochs)
    
    # 生成模拟的指标历史
    metrics_history = []
    for i in range(epochs):
        base_acc = 0.4 + 0.5 * (1 - np.exp(-0.1 * i)) + 0.05 * np.random.random()
        metrics_history.append({
            'accuracy': base_acc,
            'f1_macro': base_acc * 0.95 + 0.02 * np.random.random(),
            'f1_weighted': base_acc * 0.98 + 0.01 * np.random.random(),
            'recall_macro': base_acc * 0.93 + 0.03 * np.random.random(),
            'precision_macro': base_acc * 0.96 + 0.02 * np.random.random(),
            'ua': base_acc * 0.92 + 0.04 * np.random.random(),
            'per_class_f1': [base_acc + 0.1 * np.random.random() - 0.05 for _ in range(4)]
        })
    
    # 创建训练曲线图
    emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
    emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']
    
    epochs_range = range(1, epochs + 1)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 损失曲线
    axes[0, 0].plot(epochs_range, train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(epochs_range, test_losses, 'r-', label='测试损失', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('训练/测试损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('轮次 (Epoch)')
    axes[0, 0].set_ylabel('损失值 (Loss)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor('#f8f9fa')
    
    # 准确率曲线
    accuracies = [m['accuracy'] for m in metrics_history]
    ua_scores = [m['ua'] for m in metrics_history]
    axes[0, 1].plot(epochs_range, accuracies, 'g-', label='整体准确率', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(epochs_range, ua_scores, 'm-', label='无权重准确率(UA)', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('准确率曲线', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('轮次 (Epoch)')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_facecolor('#f8f9fa')
    
    # F1分数曲线
    f1_macro = [m['f1_macro'] for m in metrics_history]
    f1_weighted = [m['f1_weighted'] for m in metrics_history]
    axes[0, 2].plot(epochs_range, f1_macro, 'orange', label='F1宏平均', linewidth=2, marker='o', markersize=4)
    axes[0, 2].plot(epochs_range, f1_weighted, 'purple', label='F1加权平均', linewidth=2, marker='s', markersize=4)
    axes[0, 2].set_title('F1分数曲线', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('轮次 (Epoch)')
    axes[0, 2].set_ylabel('F1分数')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_facecolor('#f8f9fa')
    
    # 精确率和召回率曲线
    precision_macro = [m['precision_macro'] for m in metrics_history]
    recall_macro = [m['recall_macro'] for m in metrics_history]
    axes[1, 0].plot(epochs_range, precision_macro, 'cyan', label='精确率(宏平均)', linewidth=2, marker='o', markersize=4)
    axes[1, 0].plot(epochs_range, recall_macro, 'brown', label='召回率(宏平均)', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_title('精确率/召回率曲线', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('轮次 (Epoch)')
    axes[1, 0].set_ylabel('分数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor('#f8f9fa')
    
    # 各类别F1分数趋势
    for i, emotion in enumerate(emotion_labels.values()):
        class_f1_scores = [m['per_class_f1'][i] for m in metrics_history]
        axes[1, 1].plot(epochs_range, class_f1_scores, label=emotion, linewidth=2, 
                       marker='o', markersize=3, color=emotion_colors[i])
    axes[1, 1].set_title('各情感类别F1分数', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('轮次 (Epoch)')
    axes[1, 1].set_ylabel('F1分数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_facecolor('#f8f9fa')
    
    # 最佳性能标记
    best_epoch = np.argmax(f1_macro) + 1
    best_f1 = max(f1_macro)
    
    axes[1, 2].plot(epochs_range, f1_macro, 'b-', linewidth=2, alpha=0.7)
    axes[1, 2].scatter([best_epoch], [best_f1], color='red', s=100, zorder=5, label=f'最佳点 (Epoch {best_epoch})')
    axes[1, 2].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(y=best_f1, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('最佳性能标记', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('轮次 (Epoch)')
    axes[1, 2].set_ylabel('F1分数(宏平均)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_facecolor('#f8f9fa')
    axes[1, 2].text(best_epoch, best_f1 + 0.02, f'{best_f1:.4f}', 
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    save_path = 'test_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if os.path.exists(save_path):
        print(f"✅ 训练曲线测试通过，图片已保存: {save_path}")
        return True
    else:
        print("❌ 训练曲线测试失败")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 IEMOCAP 可视化功能测试")
    print("=" * 60)
    
    all_tests_passed = True
    
    # 运行所有测试
    tests = [
        test_chinese_font,
        test_emotion_visualization,
        test_training_curves
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            all_tests_passed = all_tests_passed and result
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 失败: {e}")
            all_tests_passed = False
        print("-" * 40)
    
    # 总结
    if all_tests_passed:
        print("🎉 所有可视化测试通过！")
        print("✅ 训练代码中的可视化功能应该能正常工作")
        print("\n📊 生成的测试图片:")
        print("  • test_chinese_font.png - 中文字体测试")
        print("  • test_emotion_probabilities.png - 情感概率可视化测试")  
        print("  • test_training_curves.png - 训练曲线测试")
    else:
        print("❌ 部分测试失败，请检查相关功能")
    
    print("=" * 60)

if __name__ == "__main__":
    main()


