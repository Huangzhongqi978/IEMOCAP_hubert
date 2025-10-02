import sys
import torch
import numpy as np
import pyaudio
import wave
import torchaudio
from PyQt5 import QtWidgets, QtCore, QtGui
from transformers import Wav2Vec2Processor, HubertModel
from models import SpeechRecognitionModel

# ====== 模型加载 ======
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad"} 
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

class Cfg:
    dropout = 0.2
    dia_layers = 2
    hidden_layer = 256
    out_class = 4
    utt_insize = 768
    attention = True
    bid_flag = False
    batch_first = False
    cuda = False

cfg = Cfg()
model = SpeechRecognitionModel(cfg)
# 直接使用torch.load，不设置weights_only参数
state_dict = torch.load("model.pkl", map_location="cpu")

model.load_state_dict(state_dict, strict=True)
model.eval()

# ====== 录音函数 ======
def record_audio(filename="temp.wav", record_seconds=3, rate=16000):
    chunk = 1024
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

# ====== 特征提取 ======
def extract_ssl_features_from_wav(wav_path, duration_sec=3):
    wav, sr = torchaudio.load(wav_path)
    target_length = int(duration_sec * sr)
    if wav.size(1) > target_length:
        wav = wav[:, :target_length]
    else:
        padding_length = target_length - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, padding_length))
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt").input_values
    return inputs

# ====== 预测 ======
def predict(model, feat_tensor):
    with torch.no_grad():
        x_try = feat_tensor.squeeze(0)
        utt_out, _ = model(x_try)
        pred_label = int(torch.argmax(utt_out, dim=1).item())
        probs = torch.softmax(utt_out, dim=-1).cpu().numpy().squeeze(0)
    return pred_label, probs

# ====== PyQt 界面 ======
class EmotionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("实时语音情感识别")
        self.resize(500, 300)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: "Microsoft YaHei";
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 10px;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 20px;
                color: #333;
            }
        """)

        # 标题
        self.title = QtWidgets.QLabel("🎤 实时语音情感识别", self)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 24px; font-weight: bold;")

        # 状态标签
        self.status = QtWidgets.QLabel("点击下方按钮开始录音", self)
        self.status.setAlignment(QtCore.Qt.AlignCenter)

        # 按钮
        self.btn_record = QtWidgets.QPushButton("🎙 开始录音并识别", self)
        self.btn_record.clicked.connect(self.record_and_predict)

        # 结果
        self.result = QtWidgets.QLabel("", self)
        self.result.setAlignment(QtCore.Qt.AlignCenter)
        self.result.setStyleSheet("font-size: 22px; font-weight: bold; color: #2E86C1;")

        # 布局
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.title)
        vbox.addSpacing(10)
        vbox.addWidget(self.status)
        vbox.addSpacing(15)
        vbox.addWidget(self.btn_record)
        vbox.addSpacing(20)
        vbox.addWidget(self.result)
        self.setLayout(vbox)

    def record_and_predict(self):
        self.status.setText("🎤 正在录音中...")
        QtWidgets.QApplication.processEvents()

        wav_path = record_audio("temp.wav", record_seconds=3)
        feats = extract_ssl_features_from_wav(wav_path)
        pred, probs = predict(model, feats)

        emotion = label_map[pred]
        confidence = probs[pred]

        self.result.setText(f"结果: {emotion} ({confidence:.2f})")
        self.status.setText("✅ 录音完成")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = EmotionApp()
    ex.show()
    sys.exit(app.exec_())
