#!/usr/bin/env python3
"""
优化版IEMOCAP情感识别训练启动脚本
目标：通过参数优化将准确率提升到70%左右

主要优化策略：
1. 增加训练轮数到8轮，充分学习
2. 降低学习率到5e-5，提高训练稳定性
3. 优化网络结构：256隐藏层，3层GRU
4. 减少冻结层到4层，允许更多参数学习
5. 添加Mixup数据增强
6. 使用余弦退火学习率调度
7. 标签平滑防止过拟合
8. 早停机制防止过训练
"""

import subprocess
import sys
import os
from datetime import datetime

def run_training():
    """运行优化版训练"""
    print("🚀 启动优化版IEMOCAP情感识别训练")
    print("=" * 60)
    
    # 检查必要文件
    required_files = [
        'train_enhanced_original.py',
        'Train_data_org.pickle',
        'models.py',
        'utils.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    print("✅ 所有必要文件检查完成")
    
    # 构建训练命令
    cmd = [
        sys.executable, 
        'train_enhanced_original.py',
        '--epochs', '8',                    # 增加训练轮数
        '--lr', '5e-5',                     # 降低学习率
        '--batch_size', '24',               # 调整批次大小
        '--hidden_layer', '256',            # 增大隐藏层
        '--dia_layers', '3',                # 增加GRU层数
        '--dropout', '0.25',                # 适度dropout
        '--freeze_layers', '4',             # 减少冻结层
        '--adversarial_weight', '0.05',     # 降低对抗权重
        '--max_grad_norm', '0.5',           # 更严格梯度裁剪
        '--mixup_alpha', '0.2',             # 启用mixup
        '--use_enhanced_gru',               # 使用增强版GRU
        '--speaker_norm',                   # 启用说话人归一化
        '--speaker_adversarial',            # 启用对抗训练
        '--exp_name', f'enhanced_gru_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ]
    
    print("🎯 训练配置:")
    print(f"   训练轮数: 8轮 (原来1轮)")
    print(f"   学习率: 5e-5 (原来1e-4)")
    print(f"   批次大小: 24 (原来32)")
    print(f"   隐藏层大小: 256 (原来128)")
    print(f"   GRU层数: 3层 (原来2层)")
    print(f"   Dropout: 0.25 (原来0.3)")
    print(f"   冻结层数: 4层 (原来6层)")
    print(f"   数据增强: Mixup (alpha=0.2)")
    print(f"   学习率调度: 余弦退火")
    print(f"   损失函数: 标签平滑交叉熵")
    print(f"   早停机制: 启用 (patience=3)")
    print(f"   🚀 GRU模型: 增强版 (EnhancedGRU)")
    print(f"      ✓ 多层残差连接")
    print(f"      ✓ 层归一化")
    print(f"      ✓ 位置编码")
    print(f"      ✓ 说话人归一化")
    print(f"      ✓ 多头自注意力")
    print(f"      ✓ 特征增强模块")
    print(f"      ✓ 对抗训练支持")
    
    print("\n🔥 开始训练...")
    print("=" * 60)
    
    try:
        # 运行训练
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ 训练完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏸️ 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        return False

def run_focal_loss_training():
    """运行使用Focal Loss的训练（如果标准训练效果不佳）"""
    print("\n🎯 启动Focal Loss优化训练")
    print("=" * 60)
    
    cmd = [
        sys.executable, 
        'train_enhanced_original.py',
        '--epochs', '8',
        '--lr', '5e-5',
        '--batch_size', '24',
        '--hidden_layer', '256',
        '--dia_layers', '3',
        '--dropout', '0.25',
        '--freeze_layers', '4',
        '--adversarial_weight', '0.05',
        '--max_grad_norm', '0.5',
        '--mixup_alpha', '0.2',
        '--use_focal_loss',                 # 启用Focal Loss
        '--focal_alpha', '0.25',
        '--focal_gamma', '2.0',
        '--exp_name', f'focal_loss_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ]
    
    print("🎯 Focal Loss配置:")
    print(f"   Focal Loss: 启用")
    print(f"   Alpha: 0.25")
    print(f"   Gamma: 2.0")
    print(f"   用途: 处理类别不平衡问题")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Focal Loss训练完成!")
        return True
    except Exception as e:
        print(f"\n❌ Focal Loss训练失败: {e}")
        return False

if __name__ == "__main__":
    print("🎵 IEMOCAP情感识别优化训练系统")
    print("目标: 将准确率从当前水平提升到70%左右")
    print("策略: 不修改架构，仅通过参数优化和训练策略改进")
    print("=" * 80)
    
    # 运行标准优化训练
    success = run_training()
    
    if success:
        print("\n🎉 优化训练完成!")
        print("\n📊 请查看results目录下的实验结果")
        print("📈 如果准确率仍未达到70%，可以尝试:")
        print("   1. 运行Focal Loss版本处理类别不平衡")
        print("   2. 进一步调整学习率和网络结构")
        print("   3. 增加更多数据增强策略")
        
        # 询问是否运行Focal Loss版本
        try:
            choice = input("\n❓ 是否运行Focal Loss版本? (y/n): ").lower().strip()
            if choice == 'y':
                run_focal_loss_training()
        except KeyboardInterrupt:
            print("\n👋 再见!")
    else:
        print("\n❌ 训练失败，请检查错误信息")
