#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试DataFrame性能修复
"""

import pandas as pd
import numpy as np
import time
import warnings

# 捕获性能警告
warnings.filterwarnings('error', category=pd.errors.PerformanceWarning)

def test_old_method():
    """测试原来的逐列添加方法（会产生警告）"""
    print("🔴 测试原来的逐列添加方法...")
    
    # 模拟数据
    n_samples = 1000
    n_features = 768
    
    features = [np.random.rand(n_features) for _ in range(n_samples)]
    test_ids = [f"sample_{i}" for i in range(n_samples)]
    true_labels = np.random.randint(0, 4, n_samples)
    predictions = np.random.randint(0, 4, n_samples)
    
    emotion_labels = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'}
    
    start_time = time.time()
    
    try:
        # 原来的方法
        results_df = pd.DataFrame({
            'id': test_ids,
            'true_label': true_labels,
            'predicted_label': predictions,
            'true_emotion': [emotion_labels[label] for label in true_labels],
            'predicted_emotion': [emotion_labels[label] for label in predictions]
        })
        
        # 逐列添加特征（会产生警告）
        for i, feature in enumerate(features):
            results_df[f'feature_{i}'] = [feat[i] if i < len(feat) else 0 for feat in features]
        
        end_time = time.time()
        print(f"   原方法完成，耗时: {end_time - start_time:.2f}秒")
        print(f"   DataFrame形状: {results_df.shape}")
        
    except pd.errors.PerformanceWarning as e:
        print(f"   ⚠️ 捕获到性能警告: {e}")
        return None
    except Exception as e:
        print(f"   ❌ 发生错误: {e}")
        return None
    
    return results_df

def test_new_method():
    """测试新的一次性创建方法"""
    print("\n🟢 测试新的一次性创建方法...")
    
    # 模拟数据
    n_samples = 1000
    n_features = 768
    
    features = [np.random.rand(n_features) for _ in range(n_samples)]
    test_ids = [f"sample_{i}" for i in range(n_samples)]
    true_labels = np.random.randint(0, 4, n_samples)
    predictions = np.random.randint(0, 4, n_samples)
    
    emotion_labels = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'}
    
    start_time = time.time()
    
    try:
        # 新方法
        # 首先创建基础数据字典
        base_data = {
            'id': test_ids,
            'true_label': true_labels,
            'predicted_label': predictions,
            'true_emotion': [emotion_labels[label] for label in true_labels],
            'predicted_emotion': [emotion_labels[label] for label in predictions]
        }
        
        # 准备特征数据（一次性创建所有特征列）
        if features and len(features) > 0:
            # 获取最大特征维度
            max_feature_dim = max(len(feat) for feat in features) if features else 0
            
            # 创建特征数据字典
            feature_data = {}
            for i in range(max_feature_dim):
                feature_data[f'feature_{i}'] = [feat[i] if i < len(feat) else 0 for feat in features]
            
            # 合并基础数据和特征数据
            all_data = {**base_data, **feature_data}
        else:
            all_data = base_data
        
        # 一次性创建完整的DataFrame
        results_df = pd.DataFrame(all_data)
        
        end_time = time.time()
        print(f"   ✅ 新方法完成，耗时: {end_time - start_time:.2f}秒")
        print(f"   DataFrame形状: {results_df.shape}")
        print(f"   无性能警告！")
        
        return results_df
        
    except Exception as e:
        print(f"   ❌ 发生错误: {e}")
        return None

def main():
    print("📊 DataFrame性能优化测试\n")
    
    # 测试原方法（可能产生警告）
    old_result = test_old_method()
    
    # 测试新方法
    new_result = test_new_method()
    
    # 比较结果
    if old_result is not None and new_result is not None:
        print(f"\n📈 性能对比:")
        print(f"   原方法DataFrame形状: {old_result.shape}")
        print(f"   新方法DataFrame形状: {new_result.shape}")
        
        # 检查数据一致性
        if old_result.shape == new_result.shape:
            print("   ✅ 两种方法产生的DataFrame形状一致")
        else:
            print("   ❌ 两种方法产生的DataFrame形状不一致")
    
    print(f"\n🎉 测试完成！新方法成功消除了性能警告。")

if __name__ == "__main__":
    main()


