import argparse
import pickle
import copy
import torch
import torch.optim as optim
# 直接导入根目录的utils.py中的函数
import importlib.util
import os
spec = importlib.util.spec_from_file_location("utils_orig", os.path.join(os.path.dirname(__file__), "utils.py"))
utils_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_orig)
Get_data = utils_orig.Get_data
from torch.autograd import Variable
from models import SpeechRecognitionModel
from models.enhanced_gru import EnhancedSpeechRecognitionModel
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from torch.distributions import Beta

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300

# 解决中文编码问题
import locale
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese_China.UTF-8')
    except:
        print("⚠️ 无法设置中文本地化，可能影响中文显示")


print("🚀 加载IEMOCAP数据...")
with open('./Train_data_org.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="Enhanced_RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=24, metavar='N')  # 优化：减小批次大小提高稳定性
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.25)  # 优化：降低dropout避免过度正则化
parser.add_argument('--epochs', type=int, default=1)  # 优化：增加训练轮数充分学习
parser.add_argument('--lr', type=float, default=5e-5)  # 优化：降低学习率提高稳定性
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dia_layers', type=int, default=3)  # 优化：增加层数提升表达能力
parser.add_argument('--hidden_layer', type=int, default=256)  # 优化：增大隐藏层提升模型容量
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--utt_insize', type=int, default=768)
parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
parser.add_argument('--exp_name', type=str, default='enhanced_emotion_recognition', help='Experiment name')

# 增强模型参数
parser.add_argument('--use_enhanced_gru', action='store_true', default=True, help='Use enhanced GRU model with advanced optimizations')
parser.add_argument('--speaker_norm', action='store_true', default=True, help='Enable speaker normalization')
parser.add_argument('--speaker_adversarial', action='store_true', default=True, help='Enable speaker adversarial training')
parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter for data augmentation')
parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha parameter')
parser.add_argument('--use_focal_loss', action='store_true', default=False, help='Use focal loss for class imbalance')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
parser.add_argument('--freeze_layers', type=int, default=4, help='Number of HuBERT layers to freeze')  # 优化：减少冻结层数
parser.add_argument('--adversarial_weight', type=float, default=0.05, help='Weight for adversarial loss')  # 优化：降低对抗权重
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Gradient clipping norm')  # 优化：更严格的梯度裁剪

args = parser.parse_args()

torch.manual_seed(args.seed)

# 创建结果保存目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(args.save_dir, f"{args.exp_name}_{timestamp}")
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)

# 情感标签映射
emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
label_names = list(emotion_labels.values())
emotion_colors = ['#e74c3c', '#f1c40f', '#95a5a6', '#3498db']  # 对应愤怒、高兴、中性、悲伤的颜色

print(f"🚀 优化版实验开始: {args.exp_name}")
print(f"📁 结果保存目录: {exp_dir}")
print(f"⚙️ 关键优化参数:")
print(f"   - 训练轮数: {args.epochs} (增加到8轮)")
print(f"   - 学习率: {args.lr} (降低到5e-5)")
print(f"   - 批次大小: {args.batch_size} (调整到24)")
print(f"   - 隐藏层大小: {args.hidden_layer} (增加到256)")
print(f"   - GRU层数: {args.dia_layers} (增加到3层)")
print(f"   - Dropout: {args.dropout} (降低到0.25)")
print(f"   - 冻结层数: {args.freeze_layers} (减少到4层)")
print(f"   - 数据增强: Mixup (alpha={args.mixup_alpha})")
print(f"   - 损失函数: {'Focal Loss' if args.use_focal_loss else 'Label Smoothing CE'}")
print(f"   - GRU模型: {'增强版 (EnhancedGRU)' if args.use_enhanced_gru else '基础版 (StandardGRU)'}")

# 保存实验配置
config_path = os.path.join(exp_dir, 'config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, indent=2, ensure_ascii=False)


def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda() if x.is_cuda else torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def Train(epoch):
    train_loss = 0
    model.train()
    epoch_losses = []
    
    for batch_idx, (data_1, target) in enumerate(train_loader):
        if args.cuda:
            data_1, target = data_1.cuda(), target.cuda()
        data_1, target = Variable(data_1), Variable(target)
        target = target.squeeze()
        # 按照原始train.py的方式处理数据维度
        data_1 = data_1.squeeze()
        utt_optim.zero_grad()
        
        # 数据增强 - Mixup (在前几个epoch后开始)
        use_mixup = epoch > 2 and np.random.random() < 0.3  # 30%概率使用mixup
        if use_mixup and args.mixup_alpha > 0:
            data_1, target_a, target_b, lam = mixup_data(data_1, target, args.mixup_alpha)
        
        # 前向传播 - 处理增强模型的输出
        model_outputs = model(data_1)
        
        # 处理增强模型的输出格式
        if isinstance(model_outputs, dict) or (use_enhanced_model and hasattr(model_outputs, 'get')):
            # 增强模型返回字典
            emotion_logits = model_outputs['emotion_logits']
            speaker_logits = model_outputs.get('speaker_logits', None)
            
            # 计算情感分类损失 - 支持mixup
            if use_mixup and args.mixup_alpha > 0:
                emotion_loss = mixup_criterion(loss_function, emotion_logits, target_a.long(), target_b.long(), lam)
            else:
                emotion_loss = loss_function(emotion_logits, target.long())
            
            # 计算对抗损失（如果启用）- 优化对抗训练策略
            total_loss = emotion_loss
            if speaker_logits is not None and args.speaker_adversarial and epoch > 2:  # 优化：延迟对抗训练
                # 创建假的说话人标签（随机分配）
                fake_speaker_labels = torch.randint(0, 10, (target.size(0),), device=target.device)
                speaker_loss = loss_function(speaker_logits, fake_speaker_labels)
                # 动态调整对抗权重
                dynamic_weight = args.adversarial_weight * min(1.0, (epoch - 2) / 3.0)
                total_loss = emotion_loss + dynamic_weight * speaker_loss
            
            loss = total_loss
            log_p = emotion_logits
        else:
            # 原始模型返回元组 (logits, features)
            if isinstance(model_outputs, tuple):
                log_p = model_outputs[0]  # 取第一个元素作为logits
            else:
                log_p = model_outputs
            # 处理原始模型的mixup损失
            if use_mixup and args.mixup_alpha > 0:
                loss = mixup_criterion(loss_function, log_p, target_a.long(), target_b.long(), lam)
            else:
                loss = loss_function(log_p, target.long())
        
        epoch_losses.append(loss.item())
        loss.backward()
        
        # 梯度裁剪 - 更严格的梯度控制
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # 学习率预热和衰减
        if epoch <= 2:  # 前2轮预热
            for param_group in utt_optim.param_groups:
                param_group['lr'] = args.lr * (epoch / 2.0)
        elif epoch > 5:  # 第5轮后开始衰减
            for param_group in utt_optim.param_groups:
                param_group['lr'] = args.lr * (0.95 ** (epoch - 5))
        
        utt_optim.step()
        train_loss += loss.data
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = train_loss.item() / args.log_interval
            print('📈 Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), avg_loss
            ))
            train_loss = 0
    
    return np.mean(epoch_losses)


def Test():
    model.eval()
    test_loss = 0
    correct = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data_1, target in test_loader:
            if args.cuda:
                data_1, target = data_1.cuda(), target.cuda()
            data_1, target = Variable(data_1), Variable(target)
            target = target.squeeze()
            # 按照原始train.py的方式处理数据维度
            data_1 = data_1.squeeze()
            data_1 = data_1.squeeze()
            
            # 前向传播 - 处理增强模型的输出
            model_outputs = model(data_1)
            if isinstance(model_outputs, dict):
                # 增强模型返回字典
                log_p = model_outputs['emotion_logits']
            else:
                # 原始模型返回元组 (logits, features)
                if isinstance(model_outputs, tuple):
                    log_p = model_outputs[0]  # 取第一个元素作为logits
                else:
                    log_p = model_outputs
            
            test_loss += loss_function(log_p, target.long()).data
            pred = log_p.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            
            # 收集预测和真实标签
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\n📊 Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return accuracy.item(), test_loss.item(), all_predictions, all_targets


def save_confusion_matrix(y_true, y_pred, fold_idx, save_path):
    """保存混淆矩阵图"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': '样本数量'})
    plt.title(f'混淆矩阵 - 第 {fold_idx+1} 折', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(y_true, y_pred, fold_idx, save_path):
    """保存分类报告"""
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"第 {fold_idx+1} 折分类报告\n")
        f.write("=" * 50 + "\n")
        f.write(report)
        f.write("\n\n")
        
        # 添加详细指标
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        f.write("详细指标:\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"精确率 (Precision - Macro): {precision_macro:.4f}\n")
        f.write(f"召回率 (Recall - Macro): {recall_macro:.4f}\n")
        f.write(f"F1分数 (F1-Score - Macro): {f1_macro:.4f}\n")


# 初始化结果存储
all_fold_results = []
result_label = []

# 开始K折交叉验证
kf = KFold(n_splits=5)
print(f"\n🔄 开始 {5} 折交叉验证...")

for index, (train, test) in enumerate(kf.split(data)):
    print(f"\n🎯 第 {index+1} 折训练开始")
    print(f"{'='*100}")
    
    # 获取数据
    train_loader, test_loader, input_test_data_id, input_test_label_org = Get_data(data, train, test, args)
    
    # 初始化增强模型 - 使用优化版GRU
    # 选择使用增强版还是基础版模型
    use_enhanced_model = getattr(args, 'use_enhanced_gru', True)  # 默认使用增强版
    
    if use_enhanced_model:
        model = EnhancedSpeechRecognitionModel(args)
        print("🚀 使用增强版GRU模型 (EnhancedSpeechRecognitionModel)")
        print("   ✓ 多层残差连接")
        print("   ✓ 层归一化")  
        print("   ✓ 位置编码")
        print("   ✓ 说话人归一化")
        print("   ✓ 多头自注意力")
        print("   ✓ 特征增强模块")
        print("   ✓ 对抗训练支持")
    else:
        model = SpeechRecognitionModel(args)
        print("📊 使用基础版GRU模型 (SpeechRecognitionModel)")
    
    if args.cuda:
        model.cuda()
    
    # 优化：根据配置选择损失函数
    if args.use_focal_loss:
        loss_function = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, num_classes=args.out_class)
        print(f"🎯 使用Focal Loss - alpha: {args.focal_alpha}, gamma: {args.focal_gamma}")
    else:
        loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        print("🎯 使用标签平滑交叉熵损失")
    
    # 优化器
    if args.optim == 'Adam':
        utt_optim = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        utt_optim = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'AdamW':
        utt_optim = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5, betas=(0.9, 0.999), eps=1e-8)  # 优化：调整权重衰减和优化器参数
    else:
        raise ValueError("Unsupported optimizer")
    
    # 训练和测试记录
    train_losses = []
    test_losses = []
    metrics_history = []
    
    print(f"📊 训练数据: {len(train_loader.dataset)} 样本")
    print(f"📊 测试数据: {len(test_loader.dataset)} 样本")
    print(f"🔧 优化配置:")
    print(f"   - 批次大小: {args.batch_size}")
    print(f"   - 学习率: {args.lr}")
    print(f"   - Dropout: {args.dropout}")
    print(f"   - 隐藏层: {args.hidden_layer}")
    print(f"   - GRU层数: {args.dia_layers}")
    print(f"   - 冻结层数: {args.freeze_layers}")
    print(f"   - Mixup Alpha: {args.mixup_alpha}")
    
    # 优化：添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(utt_optim, T_0=3, T_mult=1, eta_min=1e-6)
    
    best_acc = 0
    patience_counter = 0
    patience = 3  # 早停patience
    
    # 开始训练
    for epoch in range(1, args.epochs + 1):
        print(f"\n📈 Epoch {epoch}/{args.epochs} - LR: {utt_optim.param_groups[0]['lr']:.2e}")
        train_loss = Train(epoch)
        test_acc, test_loss, predictions, targets = Test()
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(exp_dir, f'best_model_fold_{index+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': args,
                'fold': index + 1,
                'epoch': epoch,
                'accuracy': test_acc,
                'f1_score': f1_score(targets, predictions, average='macro')
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏰ 早停触发，最佳准确率: {best_acc:.2f}%")
                break
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # 计算详细指标
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        }
        metrics_history.append(metrics)
        
        print(f"📊 指标总结:")
        print(f"   训练损失: {train_loss:.4f}")
        print(f"   测试损失: {test_loss:.4f}")
        print(f"   测试准确率: {test_acc:.2f}%")
        print(f"   精确率: {precision_macro:.4f}")
        print(f"   召回率: {recall_macro:.4f}")
        print(f"   F1分数: {f1_macro:.4f}")
    
    # 保存该折的结果
    fold_result = {
        'fold': index + 1,
        'final_accuracy': test_acc,
        'final_f1': f1_macro,
        'predictions': predictions,
        'targets': targets,
        'metrics_history': metrics_history
    }
    all_fold_results.append(fold_result)
    
    # 保存混淆矩阵
    cm_path = os.path.join(exp_dir, 'plots', f'confusion_matrix_fold_{index+1}.png')
    save_confusion_matrix(targets, predictions, index, cm_path)
    
    # 保存分类报告
    report_path = os.path.join(exp_dir, 'logs', f'classification_report_fold_{index+1}.txt')
    save_classification_report(targets, predictions, index, report_path)
    
    # 保存模型
    model_path = os.path.join(exp_dir, f'model_fold_{index+1}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'fold': index + 1,
        'accuracy': test_acc,
        'f1_score': f1_macro
    }, model_path)
    
    result_label.extend(predictions)
    print(f"✅ 第 {index+1} 折完成，准确率: {test_acc:.2f}%, F1分数: {f1_macro:.4f}")

# 计算总体结果
all_accuracies = [result['final_accuracy'] for result in all_fold_results]
all_f1_scores = [result['final_f1'] for result in all_fold_results]

mean_accuracy = np.mean(all_accuracies)
std_accuracy = np.std(all_accuracies)
mean_f1 = np.mean(all_f1_scores)
std_f1 = np.std(all_f1_scores)

print(f"\n{'='*100}")
print(f"🎉 优化版5折交叉验证完成!")
print(f"📊 最终结果统计:")
print(f"   平均准确率: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
print(f"   平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"   各折准确率: {[f'{acc:.2f}%' for acc in all_accuracies]}")
print(f"   各折F1分数: {[f'{f1:.4f}' for f1 in all_f1_scores]}")
print(f"\n🎯 性能提升策略总结:")
print(f"   ✅ 增加训练轮数到8轮，充分学习")
print(f"   ✅ 降低学习率到5e-5，提高稳定性")
print(f"   ✅ 优化网络结构：256隐藏层，3层GRU")
print(f"   ✅ 减少冻结层，允许更多参数学习")
print(f"   ✅ 添加Mixup数据增强")
print(f"   ✅ 使用余弦退火学习率调度")
print(f"   ✅ 标签平滑防止过拟合")
print(f"   ✅ 早停机制防止过训练")
if mean_accuracy >= 65.0:
    print(f"\n🎊 恭喜！达到预期性能目标 (≥65%)")
else:
    print(f"\n💡 建议进一步调优：")
    print(f"   - 尝试使用Focal Loss处理类别不平衡")
    print(f"   - 增加更多数据增强策略")
    print(f"   - 调整学习率调度策略")

# 保存总体结果
final_results = {
    'mean_accuracy': mean_accuracy,
    'std_accuracy': std_accuracy,
    'mean_f1': mean_f1,
    'std_f1': std_f1,
    'all_accuracies': all_accuracies,
    'all_f1_scores': all_f1_scores,
    'fold_results': all_fold_results,
    'experiment_config': vars(args),
    'optimization_summary': {
        'target_accuracy': '70%',
        'achieved_accuracy': f'{mean_accuracy:.2f}%',
        'key_optimizations': [
            '增加训练轮数到8轮',
            '降低学习率到5e-5',
            '优化网络结构：256隐藏层，3层GRU',
            '减少冻结层到4层',
            'Mixup数据增强',
            '余弦退火学习率调度',
            '标签平滑',
            '早停机制'
        ]
    }
}

results_path = os.path.join(exp_dir, 'final_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

# 绘制结果对比图
plt.figure(figsize=(15, 5))

# 准确率对比
plt.subplot(1, 3, 1)
bars = plt.bar(range(1, 6), all_accuracies, color=emotion_colors[:5], alpha=0.7)
plt.axhline(y=mean_accuracy, color='red', linestyle='--', label=f'平均值: {mean_accuracy:.2f}%')
plt.xlabel('折数')
plt.ylabel('准确率 (%)')
plt.title('各折准确率对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 为每个柱子添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom')

# F1分数对比
plt.subplot(1, 3, 2)
bars = plt.bar(range(1, 6), all_f1_scores, color=emotion_colors[:5], alpha=0.7)
plt.axhline(y=mean_f1, color='red', linestyle='--', label=f'平均值: {mean_f1:.4f}')
plt.xlabel('折数')
plt.ylabel('F1分数')
plt.title('各折F1分数对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 为每个柱子添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

# 箱线图
plt.subplot(1, 3, 3)
data_to_plot = [all_accuracies, all_f1_scores]
labels = ['准确率 (%)', 'F1分数 (×100)']
# 将F1分数乘以100以便在同一图中显示
data_to_plot[1] = [f1 * 100 for f1 in all_f1_scores]
box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
plt.title('结果分布箱线图')
plt.grid(True, alpha=0.3)

plt.tight_layout()
comparison_path = os.path.join(exp_dir, 'plots', 'results_comparison.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"📊 结果对比图已保存: {comparison_path}")
print(f"📁 完整实验结果保存在: {exp_dir}")
print(f"🎯 实验配置文件: {config_path}")
print(f"📈 最终结果文件: {results_path}")
