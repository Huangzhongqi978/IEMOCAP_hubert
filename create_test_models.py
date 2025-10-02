#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建测试模型文件，用于演示GUI中的模型切换功能
"""

import torch
import os
from models import SpeechRecognitionModel

class TestConfig:
    """测试模型配置"""
    dropout = 0.2
    dia_layers = 2
    hidden_layer = 256
    out_class = 4
    utt_insize = 768
    attention = True
    bid_flag = False
    batch_first = False
    cuda = False

def create_test_models():
    """创建几个测试模型文件"""
    
    # 确保results目录存在
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/test_models", exist_ok=True)
    
    config = TestConfig()
    
    # 创建基础模型
    model = SpeechRecognitionModel(config)
    
    # 保存不同版本的模型（使用随机权重作为示例）
    model_configs = [
        ("model_v1.pkl", "版本1模型"),
        ("model_v2.pth", "版本2模型"), 
        ("results/test_models/model_best.pt", "最佳模型"),
        ("results/test_models/model_epoch_10.pkl", "第10轮模型"),
        ("results/test_models/model_final.pth", "最终模型")
    ]
    
    print("🔧 正在创建测试模型文件...")
    
    for model_path, description in model_configs:
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存模型权重
            torch.save(model.state_dict(), model_path)
            print(f"✅ 已创建: {model_path} ({description})")
            
        except Exception as e:
            print(f"❌ 创建失败 {model_path}: {e}")
    
    print(f"\n🎉 测试模型创建完成！现在您可以在GUI中切换以下模型:")
    for model_path, description in model_configs:
        if os.path.exists(model_path):
            print(f"   • {model_path}")

if __name__ == "__main__":
    create_test_models()


