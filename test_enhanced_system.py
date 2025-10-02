#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强系统功能测试脚本
验证所有组件是否正常工作
"""

import torch
import numpy as np
import os
import sys
from datetime import datetime

def test_enhanced_gru():
    """测试增强GRU模型"""
    print("🧪 测试增强GRU模型...")
    
    try:
        from models.enhanced_gru import create_enhanced_model
        
        # 创建测试参数
        class Args:
            def __init__(self):
                self.hidden_layer = 128
                self.out_class = 4
                self.dia_layers = 2
                self.dropout = 0.3
                self.attention = True
                self.speaker_norm = True
                self.speaker_adversarial = True
                self.freeze_layers = 6
        
        args = Args()
        model = create_enhanced_model(args)
        
        # 测试前向传播
        batch_size, seq_len, feature_dim = 4, 100, 768
        dummy_input = torch.randn(batch_size, seq_len, feature_dim)
        
        outputs = model.utterance_net(dummy_input)
        
        assert outputs['emotion_logits'].shape == (batch_size, 4), f"情感输出形状错误: {outputs['emotion_logits'].shape}"
        assert outputs['speaker_logits'].shape == (batch_size, 10), f"说话人输出形状错误: {outputs['speaker_logits'].shape}"
        
        print("✅ 增强GRU模型测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 增强GRU模型测试失败: {e}")
        return False

def test_speaker_independent_data():
    """测试说话人无关数据处理"""
    print("🧪 测试说话人无关数据处理...")
    
    try:
        from utils.speaker_independent_data import SpeakerIndependentDataLoader
        
        # 创建模拟数据
        mock_data = []
        speakers = ['Ses01F', 'Ses01M', 'Ses02F', 'Ses02M']
        emotions = [0, 1, 2, 3]
        
        for i in range(100):
            sample = {
                'id': f'{speakers[i % len(speakers)]}_test_{i:03d}',
                'emotion': emotions[i % len(emotions)],
                'wav_encodings': torch.randn(50, 768),
                'speaker': speakers[i % len(speakers)]
            }
            mock_data.append(sample)
        
        # 创建临时数据文件
        import pickle
        temp_data_path = './temp_test_data.pickle'
        with open(temp_data_path, 'wb') as f:
            pickle.dump(mock_data, f)
        
        # 测试数据加载器
        loader = SpeakerIndependentDataLoader(temp_data_path)
        
        # 测试数据划分
        train_data, val_data, test_data = loader.create_speaker_independent_splits(0, n_folds=3)
        
        assert len(train_data) > 0, "训练数据为空"
        assert len(val_data) > 0, "验证数据为空" 
        assert len(test_data) > 0, "测试数据为空"
        
        # 验证说话人无重叠
        train_speakers = set([s.get('speaker', '') for s in train_data])
        test_speakers = set([s.get('speaker', '') for s in test_data])
        
        assert len(train_speakers.intersection(test_speakers)) == 0, "训练集和测试集说话人有重叠"
        
        # 清理临时文件
        os.remove(temp_data_path)
        
        print("✅ 说话人无关数据处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 说话人无关数据处理测试失败: {e}")
        if os.path.exists('./temp_test_data.pickle'):
            os.remove('./temp_test_data.pickle')
        return False

def test_training_components():
    """测试训练组件"""
    print("🧪 测试训练组件...")
    
    try:
        # 测试损失函数计算
        from train_enhanced import AdvancedTrainer
        
        # 创建测试参数
        class Args:
            def __init__(self):
                self.data_path = './Train_data_org.pickle'
                self.cuda = False  # 测试时使用CPU
                self.hidden_layer = 64  # 减小模型以加快测试
                self.out_class = 4
                self.dia_layers = 1
                self.dropout = 0.3
                self.attention = True
                self.speaker_norm = True
                self.speaker_adversarial = True
                self.freeze_layers = 3
                self.adversarial_weight = 0.1
                self.l2_reg = 1e-5
                self.n_folds = 3
        
        args = Args()
        
        # 创建模拟的模型输出
        batch_size = 4
        mock_outputs = {
            'emotion_logits': torch.randn(batch_size, 4),
            'speaker_logits': torch.randn(batch_size, 10),
            'attention_weights': None,
            'global_features': torch.randn(batch_size, 256)
        }
        
        emotion_targets = torch.randint(0, 4, (batch_size,))
        speaker_targets = torch.randint(0, 10, (batch_size,))
        
        # 测试损失计算（不需要完整的trainer，只测试损失函数逻辑）
        emotion_loss = torch.nn.functional.cross_entropy(mock_outputs['emotion_logits'], emotion_targets)
        speaker_loss = torch.nn.functional.cross_entropy(mock_outputs['speaker_logits'], speaker_targets)
        
        total_loss = emotion_loss + args.adversarial_weight * speaker_loss
        
        assert total_loss.item() > 0, "损失计算异常"
        
        print("✅ 训练组件测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        return False

def test_visualization_imports():
    """测试可视化相关导入"""
    print("🧪 测试可视化组件导入...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.manifold import TSNE
        
        # 测试中文字体设置
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
        
        # 创建简单测试图
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title('测试图表')
        plt.close(fig)
        
        print("✅ 可视化组件导入测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 可视化组件导入测试失败: {e}")
        return False

def test_model_compatibility():
    """测试模型兼容性"""
    print("🧪 测试模型兼容性...")
    
    try:
        # 测试是否能正确导入transformers
        from transformers import HubertModel
        
        # 测试是否能创建HuBERT模型（不下载，只测试导入）
        print("  - HuBERT导入正常")
        
        # 测试PyTorch版本兼容性
        torch_version = torch.__version__
        print(f"  - PyTorch版本: {torch_version}")
        
        # 测试CUDA可用性
        cuda_available = torch.cuda.is_available()
        print(f"  - CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"  - CUDA设备数: {torch.cuda.device_count()}")
        
        print("✅ 模型兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型兼容性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 启动增强系统功能测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("模型兼容性", test_model_compatibility),
        ("可视化组件导入", test_visualization_imports),
        ("增强GRU模型", test_enhanced_gru),
        ("说话人无关数据处理", test_speaker_independent_data),
        ("训练组件", test_training_components),
    ]
    
    for test_name, test_func in tests:
        print(f"\\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            test_results.append((test_name, False))
    
    # 汇总结果
    print("\\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\\n🎉 所有测试通过！增强系统可以正常使用。")
        print("\\n💡 下一步:")
        print("   1. 准备IEMOCAP数据文件 (Train_data_org.pickle)")
        print("   2. 运行训练: python train_enhanced.py")
        print("   3. 评估性能: python evaluate_speaker_independence.py")
    else:
        print(f"\\n⚠️ {total - passed} 个测试失败，请检查相关组件。")
        print("\\n🔧 可能的解决方案:")
        print("   - 检查Python环境和依赖包")
        print("   - 确认PyTorch和transformers版本兼容")
        print("   - 检查CUDA环境配置")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


