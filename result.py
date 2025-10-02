#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:10:17 2019

@author: jdang03
"""

import pickle
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score

with open('/home/Shi22/nas01home/Cooperation/Yongwei/IEMOCAP/IEMOCAP_hubert_base/Final_result_0.pickle', 'rb') as file:
    final_result =pickle.load(file)
with open('/home/Shi22/nas01home/Cooperation/Yongwei/IEMOCAP/IEMOCAP_hubert_base/Final_f1_0.pickle', 'rb') as file:
    Final_f1 =pickle.load(file)

print(Final_f1)

true_label = []
predict_label = []
predict_fea = []
num = 0


for i in range(len(final_result)):
    for j in range(len(final_result[i])):
        num += 1
        predict_label.append(final_result[i][j]['Predict_label'])
        true_label.append(final_result[i][j]['True_label'])
        predict_fea.append(np.array(final_result[i][j]['Predict_fea']))

print(num)

# 计算准确率和召回率
accuracy_recall = recall_score(true_label, predict_label, average='macro')
accuracy_f1 = metrics.f1_score(true_label, predict_label, average='macro')
CM_test = confusion_matrix(true_label, predict_label)

print(accuracy_recall, accuracy_f1)
print(CM_test)

# 计算WA 和UA
predict_label = np.array(predict_label)
true_label = np.array(true_label)
wa = np.mean(predict_label.astype(int) == true_label.astype(int))

predict_label_onehot = np.eye(4)[predict_label.astype(int)]
true_label_onehot = np.eye(4)[true_label.astype(int)]
ua = np.mean(np.sum((predict_label_onehot == true_label_onehot) * true_label_onehot, axis=0) / np.sum(true_label_onehot, axis=0))

print('UA={:.4f}, WA={:.4f}, F1={:.4f}'.format(ua, wa, accuracy_f1))
