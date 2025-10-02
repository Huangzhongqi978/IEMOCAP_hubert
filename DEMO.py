#!/usr/bin/env python
# coding: utf-8

# In[2]:


# --- Step 1: 读取 WAV 并提取 SSL 特征 (CPU 版) ---
import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertModel

# 配置路径
#wav_path = "./Audio/Ses01F_impro01_F000.wav"  # neutral
#wav_path = "./Audio/Ses01F_impro01_F012.wav"  # angry
#wav_path = "./Audio/Ses01F_impro02_F001.wav"  # sad
wav_path = "./Audio/Ses01F_impro03_F000.wav"  # happy
# 情感标签映射
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad"} 

# ====== 复用你原先的裁剪/填充函数（3秒） ======
def process_wav_file(wav_file, time_seconds):
    waveform, sample_rate = torchaudio.load(wav_file)
    target_length = int(time_seconds * sample_rate)
    # 裁剪或零填充到固定长度
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding_length = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))
    return waveform, sample_rate

# ====== 预处理 ======
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

def extract_ssl_features_from_my_loader(wav_path: str, duration_sec: int = 3):
    """
    读取 wav -> 3 秒裁剪/填充 -> Wav2Vec2Processor
    """
    # 1) 读取并处理到固定长度
    wav, sr = process_wav_file(wav_path, duration_sec)  # [1, N]

    # 2) processor -> inputs
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt").input_values

    return inputs

# 提取单句特征
seq_feats = extract_ssl_features_from_my_loader(wav_path, duration_sec=3)


# In[4]:


# --- Step 2: 加载训练好的模型 (CPU 版) ---
from models import SpeechRecognitionModel 

# 模型权重路径
ckpt_path = "model.pkl"

# 按你训练时的配置最小化必要字段
class Cfg:
    dropout = 0.2
    dia_layers = 2
    hidden_layer = 256
    out_class = 4
    utt_insize = 768
    attention = True
    bid_flag = False
    batch_first = False
    cuda = False   # 固定 CPU

cfg = Cfg()

model = SpeechRecognitionModel(cfg)
# 直接使用torch.load，不设置weights_only参数
state_dict = torch.load(ckpt_path, map_location="cpu")

model.load_state_dict(state_dict, strict=True)
model.eval()

print("Model loaded on CPU")


# In[58]:


# --- Step 3: 单句预测  ---
import torch
import numpy as np

def PredictOne(model, feat_tensor):
    """
    单句预测：

    返回:
      pred_label (int), probs (np.ndarray [num_classes]), logits (np.ndarray [num_classes])
    """
    model.eval()
    with torch.no_grad():
        x = feat_tensor
        x_try = x.squeeze(0)
        utt_out, hid = model(x_try)

        output = torch.argmax(utt_out, dim=1)
        pred_label = int(output.item())
        logits = utt_out.detach().cpu().numpy().squeeze(0)
        probs = torch.softmax(utt_out, dim=-1).detach().cpu().numpy().squeeze(0)

        print("Pred Label Index:", pred_label)
        if 'label_map' in globals():
            print("Pred Label Name :", label_map.get(pred_label, str(pred_label)))
        print("Logits           :", logits)
        print("Probs            :", probs)

    return pred_label, probs, logits

# ---- 调用 ----
pred_label, probs, logits = PredictOne(model, seq_feats)

