#!/usr/bin/env python 
# coding: utf-8
#如果采集的音频长度 大于 3 秒 → 只取最近 3 秒。
#如果采集的音频长度 小于 3 秒 → 前面补零，让波形强行变成 3 秒。
import sys
import os
import numpy as np
import torch
import torchaudio
import pyaudio
import wave
import threading
import queue
from datetime import datetime
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QWidget, QProgressBar, 
                             QGroupBox, QComboBox, QSlider, QSpinBox, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor
from transformers import Wav2Vec2Processor, HubertModel
from models import SpeechRecognitionModel

# 情感标签映射
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad"}
emotion_colors = {
    "neutral": "#3498db",    # 蓝色
    "happy": "#f1c40f",      # 黄色
    "angry": "#e74c3c",      # 红色
    "sad": "#9b59b6"         # 紫色
}

# --- 改进的实时音频录制类 ---
class AudioRecorder(QThread):
    data_ready = pyqtSignal(object, int)  # 发送音频数据和采样率
    
    def __init__(self, rate=16000, chunksize=1024, channels=1):
        super().__init__()
        self.rate = rate
        self.chunksize = chunksize
        self.channels = channels
        self.audio = None
        self.stream = None
        self.recording = False
        # 缓冲帧队列：每个元素为 bytes (in_data)
        # maxlen 以帧数为单位：保存最近 5 秒音频
        frames_per_second = max(1, int(self.rate / max(1, self.chunksize)))
        self.audio_buffer = deque(maxlen=frames_per_second * 5)
        # 用于保存停止录制后的完整音频（bytes）与处理后的 waveform
        self.last_audio_bytes = None
        self.last_waveform = None
    
    def init_audio(self):
        """初始化音频设备"""
        try:
            self.audio = pyaudio.PyAudio()
            return True
        except Exception as e:
            print(f"音频设备初始化失败: {e}")
            return False
    
    def start_recording(self):
        """开始录制"""
        if self.recording:
            return False
            
        if not self.audio:
            if not self.init_audio():
                return False
        
        try:
            self.recording = True
            self.audio_buffer.clear()
            self.last_audio_bytes = None
            self.last_waveform = None
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunksize,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            return True
        except Exception as e:
            print(f"开始录制失败: {e}")
            return False
    
    def stop_recording(self):
        """停止录制并保存最后的录音数据以供离线识别"""
        self.recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"关闭流时出错: {e}")
            self.stream = None
        
        # 合并缓冲区中所有数据形成最后录音（如果有）
        try:
            if len(self.audio_buffer) > 0:
                audio_data = b''.join(list(self.audio_buffer))
                self.last_audio_bytes = audio_data
                # 转为 numpy / torch waveform 并归一化
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if audio_array.size != 0:
                    waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
                    waveform = waveform / 32768.0  # 归一化
                    self.last_waveform = waveform
            else:
                self.last_audio_bytes = None
                self.last_waveform = None
        except Exception as e:
            print(f"保存最后录音时发生错误: {e}")
            self.last_audio_bytes = None
            self.last_waveform = None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.recording:
            # 将原始 bytes 存入缓冲（后续统一处理）
            self.audio_buffer.append(in_data)
            
            # 若需要实时触发（例如实时识别），可在此触发 data_ready 信号。
            # 这里仅在缓冲达到 3 秒数据量时发送一次最近3秒数据（保持原意）
            try:
                frames_needed = int(self.rate * 3 / self.chunksize)
                if len(self.audio_buffer) >= frames_needed:
                    recent_frames = list(self.audio_buffer)[-frames_needed:]
                    audio_data = b''.join(recent_frames)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
                    waveform = waveform / 32768.0  # 归一化
                    # 发送最近3秒数据（仅作为回调，不代表 GUI 必定会实时识别）
                    self.data_ready.emit(waveform, self.rate)
            except Exception as e:
                print(f"音频回调处理错误: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def get_recent_audio(self, duration_sec=3):
        """获取最近的音频数据（duration_sec 秒）"""
        if len(self.audio_buffer) == 0:
            return None, None
            
        frames_needed = max(1, int(self.rate * duration_sec / self.chunksize))
        if len(self.audio_buffer) < frames_needed:
            return None, None
            
        recent_frames = list(self.audio_buffer)[-frames_needed:]
        audio_data = b''.join(recent_frames)
        
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
            waveform = waveform / 32768.0  # 归一化
            return waveform, self.rate
        except Exception as e:
            print(f"获取音频数据错误: {e}")
            return None, None
    
    def cleanup(self):
        """清理资源"""
        self.stop_recording()
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass

# --- 音频处理函数 ---
def process_wav_data(waveform, sample_rate, time_seconds):
    """处理音频数据到固定长度（若 input 为 None 或形状不一致该函数会返回 None）"""
    try:
        if waveform is None:
            return None
        # 确保 waveform 是 2D tensor: (1, T) 或 (T,) -> 统一为 (1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3 and waveform.size(0) == 1:
            # 某些情况下 processor 返回 (1,1,T)，压缩中间维度
            waveform = waveform.squeeze(1)
        # 现在 waveform 形状应为 (1, T)
        if waveform.dim() != 2:
            return None
        
        target_length = int(time_seconds * sample_rate)
        current_length = waveform.size(1)
        
        if current_length > target_length:
            # 裁剪取最后 target_length 的数据
            start_idx = current_length - target_length
            waveform = waveform[:, start_idx:]
        elif current_length < target_length:
            # 在前面填充零
            padding_length = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (padding_length, 0))  # 在前面填充
        
        return waveform
    except Exception as e:
        print(f"音频处理错误: {e}")
        return None

# --- 特征提取工作器（保留，如需异步可启用） ---
class FeatureExtractionWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, audio_data, sample_rate, processor, duration_sec=3):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.processor = processor
        self.duration_sec = duration_sec
    
    def run(self):
        try:
            waveform = process_wav_data(self.audio_data, self.sample_rate, self.duration_sec)
            if waveform is None:
                self.error.emit("音频处理失败")
                return
            
            inputs = self.processor(waveform, sampling_rate=self.sample_rate, 
                                  return_tensors="pt").input_values
            self.finished.emit(inputs)
        except Exception as e:
            self.error.emit(f"特征提取错误: {str(e)}")

# --- 情感识别工作器（保留） ---
class EmotionRecognitionWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, features, model):
        super().__init__()
        self.features = features
        self.model = model
    
    def run(self):
        try:
            with torch.no_grad():
                if self.features.dim() == 3 and self.features.size(0) == 1:
                    x = self.features.squeeze(0)
                else:
                    x = self.features
                    
                utt_out, hid = self.model(x)
                
                pred_label = int(torch.argmax(utt_out, dim=1).item())
                emotion_name = label_map.get(pred_label, "unknown")
                probs = torch.softmax(utt_out, dim=-1).detach().cpu().numpy().squeeze(0)
                
                result = {
                    'label': pred_label,
                    'emotion': emotion_name,
                    'probabilities': probs,
                    'confidence': float(probs[pred_label]),
                    'timestamp': datetime.now()
                }
                
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"情感识别错误: {str(e)}")

# --- 主界面 ---
class EmotionRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.processor = None
        self.model = None
        # 不使用定时器做周期识别，默认提供“停止录制后离线识别”功能
        self.recognition_timer = QTimer()
        self.is_recognizing = False
        self.current_emotion = "neutral"
        self.emotion_history = deque(maxlen=20)
        self.last_waveform = None  # 存放停止录制后的最后 waveform（供离线识别）
        
        self.init_ui()
        ok = self.load_resources()
        if not ok:
            # 若加载资源失败，禁用按钮
            self.record_btn.setEnabled(False)
            self.recognize_btn.setEnabled(False)
        self.setup_connections()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("实时语音情感识别系统")
        self.setGeometry(100, 100, 900, 700)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title = QLabel("实时语音情感识别系统")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { color: #2c3e50; margin: 10px; }")
        layout.addWidget(title)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout(control_group)
        
        self.record_btn = QPushButton("开始录制")
        self.record_btn.setFont(QFont("Arial", 12))
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.recognize_btn = QPushButton("开始识别")
        self.recognize_btn.setFont(QFont("Arial", 12))
        self.recognize_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        # 初始时识别按钮不可用（没有录音）
        self.recognize_btn.setEnabled(False)
        
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.recognize_btn)
        control_layout.addStretch()
        layout.addWidget(control_group)
        
        # 状态指示器
        status_group = QGroupBox("系统状态")
        status_layout = QHBoxLayout(status_group)
        
        self.recording_status = QLabel("● 未录制")
        self.recording_status.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
        
        self.recognition_status = QLabel("● 未识别")
        self.recognition_status.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
        
        status_layout.addWidget(QLabel("录制状态:"))
        status_layout.addWidget(self.recording_status)
        status_layout.addWidget(QLabel("识别状态:"))
        status_layout.addWidget(self.recognition_status)
        status_layout.addStretch()
        layout.addWidget(status_group)
        
        # 情绪显示区域
        emotion_group = QGroupBox("情绪识别结果")
        emotion_layout = QVBoxLayout(emotion_group)
        
        self.emotion_label = QLabel("等待识别...")
        self.emotion_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setMinimumHeight(120)
        self.emotion_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 3px solid #bdc3c7;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        self.confidence_label = QLabel("置信度: 0.0%")
        self.confidence_label.setFont(QFont("Arial", 14))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        
        emotion_layout.addWidget(self.emotion_label)
        emotion_layout.addWidget(self.confidence_label)
        layout.addWidget(emotion_group)
        
        # 概率显示
        prob_group = QGroupBox("情绪概率分布")
        prob_layout = QHBoxLayout(prob_group)
        
        self.prob_bars = {}
        self.prob_labels = {}
        
        for emotion_id, emotion_name in label_map.items():
            prob_widget = QWidget()
            prob_layout_widget = QVBoxLayout(prob_widget)
            
            label = QLabel(emotion_name.upper())
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Arial", 10, QFont.Bold))
            
            prob_value = QLabel("0%")
            prob_value.setAlignment(Qt.AlignCenter)
            prob_value.setFont(QFont("Arial", 10))
            
            prob_bar = QProgressBar()
            prob_bar.setMinimum(0)
            prob_bar.setMaximum(100)
            prob_bar.setValue(0)
            prob_bar.setTextVisible(False)
            prob_bar.setMinimumHeight(20)
            
            prob_layout_widget.addWidget(label)
            prob_layout_widget.addWidget(prob_value)
            prob_layout_widget.addWidget(prob_bar)
            prob_layout.addWidget(prob_widget)
            
            self.prob_bars[emotion_name] = prob_bar
            self.prob_labels[emotion_name] = prob_value
        
        layout.addWidget(prob_group)
        
        # 历史记录
        history_group = QGroupBox("识别历史")
        history_layout = QVBoxLayout(history_group)
        
        self.history_text = QTextEdit()
        self.history_text.setMaximumHeight(120)
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont("Arial", 9))
        
        history_layout.addWidget(self.history_text)
        layout.addWidget(history_group)
        
        # 状态栏
        self.status_label = QLabel("系统就绪")
        self.statusBar().addWidget(self.status_label)
        
    def load_resources(self):
        """加载模型和处理器"""
        try:
            self.status_label.setText("正在加载音频处理器...")
            QApplication.processEvents()
            
            # 加载处理器
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            
            self.status_label.setText("正在加载情感识别模型...")
            QApplication.processEvents()
            
            # 加载模型
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
            self.model = SpeechRecognitionModel(cfg)
            
            # 检查模型文件是否存在
            if not os.path.exists("model.pkl"):
                QMessageBox.critical(self, "错误", "未找到模型文件 model.pkl")
                return False
                
            state_dict = torch.load("model.pkl", map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            
            self.status_label.setText("资源加载完成 - 点击'开始录制'开始")
            return True
            
        except Exception as e:
            error_msg = f"加载失败: {str(e)}"
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            return False
    
    def setup_connections(self):
        """设置信号连接"""
        self.record_btn.clicked.connect(self.toggle_recording)
        self.recognize_btn.clicked.connect(self.toggle_recognition)
        # 保留 data_ready 信号连接：如果未来需要实时识别可以在回调中调用 self.extract_and_recognize
        self.recorder.data_ready.connect(self.on_audio_data_ready)
    
    def toggle_recording(self):
        """切换录制状态：开始 -> 停止"""
        if not self.recorder.recording:
            # 开始录制
            success = self.recorder.start_recording()
            if success:
                self.record_btn.setText("停止录制")
                self.record_btn.setStyleSheet("QPushButton { background-color: #c0392b; color: white; }")
                # 录制开始后 **禁用离线识别按钮**（避免在录制过程中点击识别）
                self.recognize_btn.setEnabled(False)
                self.recording_status.setText("● 录制中")
                self.recording_status.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")
                self.status_label.setText("录制中... 录制停止后可进行离线识别")
            else:
                QMessageBox.warning(self, "警告", "无法启动音频录制，请检查麦克风设备")
        else:
            # 停止录制
            self.recorder.stop_recording()
            # 从 recorder 中获取最后一次录音 waveform（若有）
            self.last_waveform = self.recorder.last_waveform
            # 停止录制后允许离线识别（若存在录音）
            self.recognize_btn.setEnabled(self.last_waveform is not None)
            
            self.record_btn.setText("开始录制")
            self.record_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; }")
            self.recording_status.setText("● 未录制")
            self.recording_status.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
            self.status_label.setText("录制已停止，可进行离线识别")
            
            # 如果之前处于识别状态（本设计中识别为单次离线模式，不需要额外处理）
            # 若有实时识别逻辑，请在此处停止
            self.is_recognizing = False
    
    def toggle_recognition(self):
        """触发一次离线识别（仅在未录制时，对最后录音进行识别）"""
        # 如果当前仍在录制，则禁止触发离线识别
        if self.recorder.recording:
            QMessageBox.warning(self, "警告", "当前正在录制，无法进行离线识别。请先停止录制。")
            return
        
        # 确保存在上一次录音 waveform
        if self.last_waveform is None:
            QMessageBox.warning(self, "警告", "未检测到最近的录音，请先录制音频后再尝试识别。")
            return
        
        # 禁用识别按钮以防重复点击
        self.recognize_btn.setEnabled(False)
        self.status_label.setText("识别中，请稍候...")
        QApplication.processEvents()
        
        try:
            # 处理并识别（同步在主线程）
            processed_waveform = process_wav_data(self.last_waveform, self.recorder.rate, 3)
            if processed_waveform is None:
                QMessageBox.warning(self, "错误", "音频处理失败，无法识别。")
                self.recognize_btn.setEnabled(True)
                return
            
            inputs = self.processor(processed_waveform, sampling_rate=self.recorder.rate, 
                                  return_tensors="pt").input_values
            
            with torch.no_grad():
                if inputs.dim() == 3 and inputs.size(0) == 1:
                    x = inputs.squeeze(0)
                else:
                    x = inputs
                    
                utt_out, hid = self.model(x)
                
                pred_label = int(torch.argmax(utt_out, dim=1).item())
                emotion_name = label_map.get(pred_label, "unknown")
                probs = torch.softmax(utt_out, dim=-1).detach().cpu().numpy().squeeze(0)
                confidence = float(probs[pred_label])
                
                result = {
                    'label': pred_label,
                    'emotion': emotion_name,
                    'probabilities': probs,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }
                
                self.update_display(result)
                self.status_label.setText(f"离线识别完成: {emotion_name} ({confidence:.2%})")
        except Exception as e:
            self.status_label.setText(f"识别错误: {str(e)}")
        finally:
            # 重新启用识别按钮（若仍存在上次录音）
            self.recognize_btn.setEnabled(self.last_waveform is not None)
    
    def process_audio(self):
        """留作周期或实时识别的入口（当前未被定时器使用）"""
        if self.recorder.recording:
            # 示例：获取最近3秒并进行识别（若需要实时识别可打开）
            waveform, sample_rate = self.recorder.get_recent_audio(duration_sec=3)
            if waveform is not None:
                self.extract_and_recognize(waveform, sample_rate)
    
    def on_audio_data_ready(self, waveform, sample_rate):
        """音频数据就绪信号（用于实时识别场景，当前未自动触发）"""
        # 保留接口：如果你希望在录制过程中实时识别，可以取消下面注释并调整逻辑
        # if not self.recorder.recording:
        #     return
        # self.extract_and_recognize(waveform, sample_rate)
        pass
    
    def extract_and_recognize(self, waveform, sample_rate):
        """提取特征并识别情感（与离线识别重复的核心逻辑）"""
        try:
            processed_waveform = process_wav_data(waveform, sample_rate, 3)
            if processed_waveform is None:
                return
                
            inputs = self.processor(processed_waveform, sampling_rate=sample_rate, 
                                  return_tensors="pt").input_values
            
            with torch.no_grad():
                if inputs.dim() == 3 and inputs.size(0) == 1:
                    x = inputs.squeeze(0)
                else:
                    x = inputs
                    
                utt_out, hid = self.model(x)
                
                pred_label = int(torch.argmax(utt_out, dim=1).item())
                emotion_name = label_map.get(pred_label, "unknown")
                probs = torch.softmax(utt_out, dim=-1).detach().cpu().numpy().squeeze(0)
                confidence = float(probs[pred_label])
                
                result = {
                    'label': pred_label,
                    'emotion': emotion_name,
                    'probabilities': probs,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }
                
                self.update_display(result)
                
        except Exception as e:
            self.status_label.setText(f"识别错误: {str(e)}")
    
    def update_display(self, result):
        """更新显示"""
        emotion = result['emotion']
        confidence = result['confidence']
        
        # 更新当前情绪
        self.current_emotion = emotion
        self.emotion_history.append(result)
        
        # 更新情绪显示
        color = emotion_colors.get(emotion, "#95a5a6")
        self.emotion_label.setText(emotion.upper())
        self.emotion_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color}20;
                border: 3px solid {color};
                border-radius: 10px;
                padding: 20px;
                color: {color};
            }}
        """)
        
        self.confidence_label.setText(f"置信度: {confidence:.2%}")
        self.confidence_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")
        
        # 更新概率条
        for emotion_name, prob_bar in self.prob_bars.items():
            try:
                idx = list(label_map.values()).index(emotion_name)
                prob = result['probabilities'][idx] * 100
                prob_bar.setValue(int(prob))
                color_e = emotion_colors.get(emotion_name, "#95a5a6")
                prob_bar.setStyleSheet(f"""
                    QProgressBar {{
                        border: 1px solid grey;
                        border-radius: 3px;
                        text-align: center;
                        background-color: #ecf0f1;
                    }}
                    QProgressBar::chunk {{
                        background-color: {color_e};
                        border-radius: 2px;
                    }}
                """)
                self.prob_labels[emotion_name].setText(f"{prob:.1f}%")
            except Exception:
                # 若索引不匹配则跳过
                continue
        
        # 更新历史记录
        timestamp = result['timestamp'].strftime("%H:%M:%S")
        history_entry = f"[{timestamp}] {emotion} ({confidence:.2%})"
        self.history_text.append(history_entry)
        
        # 限制历史记录长度
        if self.history_text.document().lineCount() > 50:
            cursor = self.history_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        
        self.status_label.setText(f"识别完成: {emotion} ({confidence:.2%})")
    
    def closeEvent(self, event):
        """关闭事件"""
        try:
            self.recognition_timer.stop()
        except:
            pass
        if self.recorder.recording:
            self.recorder.stop_recording()
        self.recorder.cleanup()
        event.accept()

def main():
    # 检查依赖
    try:
        import pyaudio
    except ImportError:
        print("请安装pyaudio: pip install pyaudio")
        return
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = EmotionRecognitionGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
