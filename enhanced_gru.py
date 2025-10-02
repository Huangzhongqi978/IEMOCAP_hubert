#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的GRU模型架构 - 针对跨说话人情感识别优化
包含说话人归一化、对抗训练、增强特征提取等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import math

class GradientReversalLayer(Function):
    """梯度反转层 - 用于说话人对抗训练"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def gradient_reverse(x, alpha=1.0):
    """梯度反转函数"""
    return GradientReversalLayer.apply(x, alpha)

class AdaptiveInstanceNormalization(nn.Module):
    """自适应实例归一化 - 说话人归一化层"""
    
    def __init__(self, num_features, eps=1e-5):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # 可学习的缩放和偏移参数
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, num_features]
        """
        # 计算实例级别的均值和方差（跨序列维度）
        mean = x.mean(dim=1, keepdim=True)  # [batch_size, 1, num_features]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [batch_size, 1, num_features]
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习的仿射变换
        out = x_norm * self.weight.unsqueeze(0).unsqueeze(0) + self.bias.unsqueeze(0).unsqueeze(0)
        
        return out

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        return context, attention_weights
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # 线性变换得到Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 输出投影
        output = self.w_o(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        pos_encoding = self.pe[:seq_len, :].transpose(0, 1)  # [1, seq_len, d_model]
        return x + pos_encoding

class EnhancedGRUModel(nn.Module):
    """增强的GRU模型 - 针对跨说话人情感识别优化"""
    
    def __init__(self, input_size, hidden_size, output_size, args):
        super(EnhancedGRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = args.dia_layers
        self.dropout_rate = args.dropout
        self.use_attention = args.attention
        self.use_speaker_norm = getattr(args, 'speaker_norm', True)
        self.use_adversarial = getattr(args, 'speaker_adversarial', True)
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size * 2)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_size * 2)
        
        # 说话人归一化层
        if self.use_speaker_norm:
            self.speaker_norm = AdaptiveInstanceNormalization(hidden_size * 2)
        
        # 双向GRU层
        self.gru_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_dim = hidden_size * 2 if i == 0 else hidden_size * 4
            self.gru_layers.append(
                nn.GRU(input_dim, hidden_size * 2, batch_first=True, 
                      bidirectional=True, dropout=self.dropout_rate if i < self.num_layers-1 else 0)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size * 4))
        
        # 多头自注意力
        if self.use_attention:
            self.self_attention = MultiHeadSelfAttention(
                d_model=hidden_size * 4, 
                num_heads=8, 
                dropout=self.dropout_rate
            )
        
        # 特征增强模块
        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )
        
        # 全局池化策略
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        
        # 情感分类头
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),  # avg + max pooling
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 说话人分类头（用于对抗训练）
        if self.use_adversarial:
            self.speaker_classifier = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_size // 2, 10)  # IEMOCAP有10个说话人
            )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, alpha=1.0):
        """
        前向传播
        x: [batch_size, seq_len, input_size]
        alpha: 梯度反转强度（用于对抗训练）
        """
        batch_size, seq_len, _ = x.size()
        
        # 输入投影和位置编码
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_size*2]
        x = self.pos_encoding(x)
        
        # 说话人归一化
        if self.use_speaker_norm:
            x = self.speaker_norm(x)
        
        x = self.dropout(x)
        
        # 多层双向GRU
        for i, (gru_layer, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            residual = x if i > 0 else None
            
            gru_out, _ = gru_layer(x)  # [batch_size, seq_len, hidden_size*4]
            gru_out = layer_norm(gru_out)
            
            # 残差连接（从第二层开始）
            if residual is not None and residual.size(-1) == gru_out.size(-1):
                gru_out = gru_out + residual
            
            x = self.dropout(gru_out)
        
        # 多头自注意力
        attention_weights = None
        if self.use_attention:
            x, attention_weights = self.self_attention(x)
        
        # 特征增强
        enhanced_features = self.feature_enhancement(x)  # [batch_size, seq_len, hidden_size*2]
        
        # 拼接原始特征和增强特征
        combined_features = torch.cat([x, enhanced_features], dim=-1)  # [batch_size, seq_len, hidden_size*6]
        
        # 降维到统一大小
        combined_features = combined_features[:, :, :self.hidden_size*4]  # [batch_size, seq_len, hidden_size*4]
        
        # 全局池化
        # 转置用于池化操作
        pooling_input = combined_features.transpose(1, 2)  # [batch_size, hidden_size*4, seq_len]
        
        avg_pooled = self.global_pooling(pooling_input).squeeze(-1)  # [batch_size, hidden_size*4]
        max_pooled = self.global_max_pooling(pooling_input).squeeze(-1)  # [batch_size, hidden_size*4]
        
        # 拼接平均池化和最大池化结果
        global_features = torch.cat([avg_pooled, max_pooled], dim=-1)  # [batch_size, hidden_size*8]
        
        # 降维到统一大小
        global_features = global_features[:, :self.hidden_size*4]  # [batch_size, hidden_size*4]
        
        # 情感分类
        emotion_logits = self.emotion_classifier(global_features)
        
        # 说话人对抗分类
        speaker_logits = None
        if self.use_adversarial:
            reversed_features = gradient_reverse(global_features, alpha)
            speaker_logits = self.speaker_classifier(reversed_features)
        
        return {
            'emotion_logits': emotion_logits,
            'speaker_logits': speaker_logits,
            'attention_weights': attention_weights,
            'global_features': global_features
        }

class EnhancedSpeechRecognitionModel(nn.Module):
    """增强的语音识别模型主类"""
    
    def __init__(self, args):
        super(EnhancedSpeechRecognitionModel, self).__init__()
        
        # HuBERT特征提取器（冻结部分层以减少过拟合）
        from transformers import HubertModel
        self.feature_extractor = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        
        # 冻结前几层
        freeze_layers = getattr(args, 'freeze_layers', 6)
        if freeze_layers > 0:
            for i, layer in enumerate(self.feature_extractor.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # 特征维度适配
        hubert_dim = self.feature_extractor.config.hidden_size  # 768
        
        # 增强的GRU模型
        self.utterance_net = EnhancedGRUModel(
            input_size=hubert_dim,
            hidden_size=args.hidden_layer,
            output_size=args.out_class,
            args=args
        )
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, input_waveform, alpha=1.0):
        """
        前向传播
        input_waveform: 音频波形输入
        alpha: 对抗训练强度
        """
        # HuBERT特征提取
        with torch.no_grad() if hasattr(self, '_freeze_hubert') else torch.enable_grad():
            hubert_features = self.feature_extractor(input_waveform).last_hidden_state
        
        hubert_features = self.dropout(hubert_features)
        
        # 通过增强GRU网络
        outputs = self.utterance_net(hubert_features, alpha=alpha)
        
        return outputs

def create_enhanced_model(args):
    """创建增强模型的工厂函数"""
    
    # 设置默认参数
    if not hasattr(args, 'speaker_norm'):
        args.speaker_norm = True
    if not hasattr(args, 'speaker_adversarial'):
        args.speaker_adversarial = True
    if not hasattr(args, 'freeze_layers'):
        args.freeze_layers = 6
    
    model = EnhancedSpeechRecognitionModel(args)
    
    print("🚀 增强模型创建成功!")
    print(f"   - 说话人归一化: {'✓' if args.speaker_norm else '✗'}")
    print(f"   - 对抗训练: {'✓' if args.speaker_adversarial else '✗'}")
    print(f"   - 冻结HuBERT层数: {args.freeze_layers}")
    print(f"   - 注意力机制: {'✓' if args.attention else '✗'}")
    
    return model

if __name__ == "__main__":
    # 测试模型
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
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"情感分类输出形状: {outputs['emotion_logits'].shape}")
    if outputs['speaker_logits'] is not None:
        print(f"说话人分类输出形状: {outputs['speaker_logits'].shape}")
    print("✅ 模型测试通过!")


