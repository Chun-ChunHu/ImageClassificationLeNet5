# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:16:06 2023

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:39:32 2023

@author: user
"""
# Load packages
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm 
from keras import layers
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from model.network import LeNet

# Path set
root_path = 'C:\\Hu\\0_IMIS\\4_course\\DeepLearning\\'
data_path = root_path + 'data\\'

# Split required data and decide model input size 
train, test, val = ([],[],[])
train_images, train_labels, test_images, test_labels, val_images, val_labels = ([],[],[],[],[],[])
data_type = ['train', 'test', 'val']
column = 1
size = 64

# Read category from txt file
for i in range(len(data_type)):
    globals()[data_type[i]] = np.genfromtxt(data_path + data_type[i] + '.txt', delimiter=' ', dtype = 'str', skip_header=1)
    # globals()[data_type[i]] = list(filter(lambda x: int(x[1]) <= 4, globals()[data_type[i]]))                            
        
def FeatureExtractorGrayscale(img, size, column):
        feature = []
        img = cv2.imread(data_path + globals()[data_type[i]][j][0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(column, size, size)
        # Normalize feature vector
        gray = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return gray  
    
def FeatureExtractorHIST(img, size):
        feature = []
        img = cv2.imread(data_path + globals()[data_type[i]][j][0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Compute color histograms for each channel
        h = cv2.calcHist([hsv], [0], None, [256], [0, 256])
        s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        # Concatenate histograms into a single feature vector
        hist = np.concatenate([h, s, v]).flatten()
        # Normalize feature vector
        hist /= np.sum(hist)
        return hist
            
# Split images and labels
for i in range(len(data_type)):
    for j in tqdm(range(len(globals()[data_type[i]]))):
        img = FeatureExtractorGrayscale(globals()[data_type[i]][j][0], size, column)
        label = globals()[data_type[i]][j][1]
        globals()[data_type[i] + '_images'].append(img)
        globals()[data_type[i] + '_labels'].append(int(label)) 
    globals()[data_type[i] + '_images'] = np.array(globals()[data_type[i] + '_images']).reshape(len(globals()[data_type[i] + '_images']), column, size, size)
    globals()[data_type[i] + '_labels'] = np.eye(len(np.unique(globals()[data_type[i] + '_labels'])), dtype = np.int16)[globals()[data_type[i] + '_labels']]
        
# Function for caculating TOP-1, TOP-5 score of prediction
def top1_5_score(test_labels, prediction, model):
    # label = model.classes
    label = np.arange(0, 50, 1)
    top1_score = 0
    top5_score = 0
    test_labels = np.argmax(test_labels, axis = -1)
    for i in tqdm(range(len(test_labels))):
        top5_ans = np.argpartition(prediction[i], -5)[-5:]
        if int(test_labels[i]) in label[top5_ans]:
            top5_score = top5_score + 1
        if int(test_labels[i]) == label[np.argmax(prediction[i])]:
            top1_score = top1_score + 1
    # print(top1_score/len(test_labels) , top5_score/len(test_labels))
    return top1_score/len(test_labels) , top5_score/len(test_labels)    

# Shuffle training data to balance the input data in each batch
train_images, train_labels = shuffle(train_images, train_labels, random_state = 0)

# Call the LeNet obj from */model/network
net = LeNet()
# Train network, parameters included (training_data, training_label, batch_size, epoch, weights_file)
net.train(train_images, train_labels, 256, 1, 'model_weights.pkl')
net.predict(test_images, test_labels, 449)


# Load model saved previously
import pickle

with open('3x3_model_weights.pkl', 'rb') as f:
    model, loss, auc = pickle.load(f)
    
# Plot auc and loss   
plt.plot(auc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['accuracy'], loc='upper right')
plt.show()   
    