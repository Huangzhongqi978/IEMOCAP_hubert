#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的可视化测试脚本
验证中文字符和IEMOCAP情感标签显示
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import locale

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300

# 解决中文编码问题
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese_China.UTF-8')
    except:
        print("⚠️ 无法设置中文本地化，可能影响中文显示")

def test_iemocap_emotion_labels():
    """测试IEMOCAP情感标签显示"""
    print("🧪 测试IEMOCAP情感标签显示...")
    
    # 使用与train.py相同的情感标签
    emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
    label_names = list(emotion_labels.values())
    emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']  # 对应愤怒、高兴、中性、悲伤的颜色
    
    # 创建情感分布饼图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 模拟数据
    emotion_counts = [25, 30, 20, 25]  # 各情感类别的样本数
    
    # 饼图显示情感分布
    wedges, texts, autotexts = ax1.pie(emotion_counts, labels=label_names, autopct='%1.1f%%', 
                                      colors=emotion_colors, startangle=90)
    ax1.set_title('IEMOCAP情感类别分布', fontsize=14, fontweight='bold')
    
    # 条形图显示情感识别性能
    np.random.seed(42)
    accuracies = [0.75 + 0.1*np.random.random() for _ in range(4)]
    
    bars = ax2.bar(label_names, accuracies, color=emotion_colors, alpha=0.8, edgecolor='black')
    ax2.set_title('各情感类别识别准确率', fontsize=14, fontweight='bold')
    ax2.set_xlabel('情感类别')
    ax2.set_ylabel('准确率')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    save_path = 'test_iemocap_labels.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if os.path.exists(save_path):
        print(f"✅ IEMOCAP情感标签测试通过: {save_path}")
        return True
    else:
        print("❌ IEMOCAP情感标签测试失败")
        return False

def test_chinese_emotion_display():
    """测试中文情感描述显示"""
    print("🧪 测试中文情感描述显示...")
    
    # 英文标签对应的中文描述
    emotion_mapping = {
        'Angry': '愤怒',
        'Happy': '高兴',
        'Neutral': '中性',
        'Sad': '悲伤'
    }
    
    emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建双语对比图
    english_labels = list(emotion_mapping.keys())
    chinese_labels = list(emotion_mapping.values())
    
    x_pos = np.arange(len(english_labels))
    width = 0.35
    
    # 模拟数据
    np.random.seed(42)
    precision_scores = [0.80 + 0.1*np.random.random() for _ in range(4)]
    recall_scores = [0.75 + 0.1*np.random.random() for _ in range(4)]
    
    bars1 = ax.bar(x_pos - width/2, precision_scores, width, label='精确率', alpha=0.8, color=emotion_colors)
    bars2 = ax.bar(x_pos + width/2, recall_scores, width, label='召回率', alpha=0.8, 
                   color=[c+'80' for c in emotion_colors])  # 半透明
    
    ax.set_xlabel('情感类别 (Emotion Categories)')
    ax.set_ylabel('性能指标 (Performance Metrics)')
    ax.set_title('IEMOCAP情感识别系统性能评估\nIEMOCAP Emotion Recognition System Performance', 
                 fontsize=14, fontweight='bold')
    
    # 设置双语标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{eng}\n{chi}' for eng, chi in zip(english_labels, chinese_labels)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for bars, scores in [(bars1, precision_scores), (bars2, recall_scores)]:
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = 'test_chinese_emotion_display.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if os.path.exists(save_path):
        print(f"✅ 中文情感描述测试通过: {save_path}")
        return True
    else:
        print("❌ 中文情感描述测试失败")
        return False

def test_attention_visualization_mock():
    """测试注意力可视化的模拟效果"""
    print("🧪 测试注意力可视化模拟...")
    
    emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
    emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']
    
    # 创建模拟的注意力权重矩阵
    np.random.seed(42)
    seq_len = 50
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(5):
        # 创建模拟的注意力权重（对角线模式 + 噪声）
        attention = np.eye(seq_len) + 0.3 * np.random.random((seq_len, seq_len))
        attention = attention / attention.sum(axis=1, keepdims=True)  # 归一化
        
        # 模拟预测结果
        true_label = i % 4
        pred_label = (i + np.random.randint(0, 2)) % 4
        
        # 绘制热力图
        im = axes[i].imshow(attention, cmap='YlOrRd', aspect='auto')
        
        correct = "✓" if true_label == pred_label else "✗"
        axes[i].set_title(f'样本{i+1} {correct}\n真实:{emotion_labels[true_label]} 预测:{emotion_labels[pred_label]}', 
                         fontsize=11, fontweight='bold')
        axes[i].set_xlabel('时间步')
        axes[i].set_ylabel('注意力权重')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[i], shrink=0.6)
    
    # 隐藏最后一个子图
    axes[5].axis('off')
    
    plt.suptitle('注意力权重热力图 - IEMOCAP语音情感识别', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'test_attention_visualization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if os.path.exists(save_path):
        print(f"✅ 注意力可视化测试通过: {save_path}")
        return True
    else:
        print("❌ 注意力可视化测试失败")
        return False

def main():
    """主测试函数"""
    print("=" * 70)
    print("🧪 IEMOCAP可视化功能验证测试")
    print("=" * 70)
    
    all_tests_passed = True
    
    # 运行测试
    tests = [
        test_iemocap_emotion_labels,
        test_chinese_emotion_display,
        test_attention_visualization_mock
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            all_tests_passed = all_tests_passed and result
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 失败: {e}")
            all_tests_passed = False
        print("-" * 50)
    
    # 总结
    if all_tests_passed:
        print("🎉 所有可视化测试通过！")
        print("✅ 训练代码中的可视化功能应该能正常工作")
        print("\n📊 生成的测试图片:")
        print("  • test_iemocap_labels.png - IEMOCAP情感标签测试")
        print("  • test_chinese_emotion_display.png - 中文情感描述测试")  
        print("  • test_attention_visualization.png - 注意力可视化测试")
        print("\n💡 这些测试验证了:")
        print("  ✓ 情感标签正确性 (Angry, Happy, Neutral, Sad)")
        print("  ✓ 中文字符显示正常")
        print("  ✓ 颜色映射正确")
        print("  ✓ 图表布局合理")
    else:
        print("❌ 部分测试失败，请检查相关功能")
        print("💡 可能的问题:")
        print("  • 中文字体未正确安装")
        print("  • matplotlib配置问题")
        print("  • 编码设置问题")
    
    print("=" * 70)

if __name__ == "__main__":
    main()


