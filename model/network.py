import numpy as np
import pandas as pd
import pickle 
import sys
from tqdm import tqdm
from time import *
from model.loss import *
from model.layers import *


class LeNet:
    def __init__(self):
        # Architecture of models
        #=================================================================================================================
        # Lenet-5                                                     # LeNet-5 + convLayer
        # input: 64x64x1                                              # input: 64x64x1
        # conv1: (5x5x6)@s1p2 -> 60x60x6 {(64-5)/1+1}                 # conv1: (3x3x3)@s1p2 -> 62x62x3 {(64-3)/1+1}
        # maxpool2: (2x2)@s2 -> 30x30x6 {(60-2)/2+1}                  # conv2: (3x3x6)@s1p2 -> 60x60x6 {(62-3)/1+1}
        # conv3: (5x5x16)@s1p0 -> 26x26x16 {(30-5)/1+1}               # maxpool3: (2x2)@s2 -> 30x30x6 {(60-2)/2+1}
        # maxpool4: (2x2)@s2 -> 13x13x16 {(26-2)/2+1}                 # conv4: (3x3x16)@s1p0 -> 28x28x16 {(30-3)/1+1}
        # conv5: (5x5x120)@s1p0 -> 9x9x120 {(13-5)/1+1}               # maxpool5: (2x2)@s2 -> 14x14x16 {(28-2)/2+1}
        # fc6: 9720 -> 960                                            # conv6: (3x3x120)@s1p0 -> 12x12x120 {(14-3)/1+1} 
        # fc7: 960 -> 50                                              # fc7: 17280 -> 1200
        # softmax: 50 -> 50                                           # fc8: 1200 -> 50
        #                                                             # softmax: 50 -> 50
        #=================================================================================================================
        lr = 0.01
        kernel_size = 3 # set kernel_size = 3 to meet the requirement of the homework
        # set the precision to a high value to avoid underflow
        self.layers = []
        # additional conv layer
        self.layers.append(Convolution2D(inputs_channel = 1, num_filters = 3, kernel_size = kernel_size, padding = 0, stride = 1, learning_rate = lr, name = 'conv1'))
        self.layers.append(Convolution2D(inputs_channel = 3, num_filters = 6, kernel_size = kernel_size, padding = 0, stride = 1, learning_rate = lr, name = 'conv2'))
        self.layers.append(Swish()) # replace with Custom_Sigmoid for using activation function = x*sigmoid(x)
        self.layers.append(Maxpooling2D(pool_size = 2, stride = 2, name = 'maxpool3'))
        self.layers.append(Convolution2D(inputs_channel = 6, num_filters = 16, kernel_size = kernel_size, padding = 0, stride = 1, learning_rate = lr, name = 'conv4'))
        self.layers.append(Swish()) # replace with Custom_Sigmoid for using activation function = x*sigmoid(x)
        self.layers.append(Maxpooling2D(pool_size = 2, stride = 2, name = 'maxpool5'))
        self.layers.append(Convolution2D(inputs_channel = 16, num_filters = 120, kernel_size = kernel_size, padding = 0, stride = 1, learning_rate = lr, name = 'conv5'))
        self.layers.append(Swish()) # replace with Custom_Sigmoid for using activation function = x*sigmoid(x)
        self.layers.append(Flatten())
        self.layers.append(FullyConnected(num_inputs = 17280, num_outputs = 1200, learning_rate = lr, name = 'fc6'))
        self.layers.append(Swish()) # replace with Custom_Sigmoid for using activation function = x*sigmoid(x)
        self.layers.append(FullyConnected(num_inputs = 1200, num_outputs = 50, learning_rate = lr, name = 'fc7'))
        self.layers.append(Softmax())
        self.lay_num = len(self.layers)
        
    def train(self, training_data, training_label, batch_size, epoch, weights_file):
        total_acc = 0
        loss = np.array([], dtype = np.float32)
        auc = np.array([], dtype = np.float32)
        for e in range(epoch):
            for batch_index in tqdm(range(0, training_data.shape[0], batch_size)):
                # batch input
                if batch_index + batch_size < training_data.shape[0]:
                    data = training_data[batch_index:batch_index+batch_size]
                    label = training_label[batch_index:batch_index + batch_size]
                else:
                    data = training_data[batch_index:training_data.shape[0]]
                    label = training_label[batch_index:training_label.shape[0]]

                temp_loss = 0
                temp_acc = 0
                
                for b in range(data.shape[0]):
                    x = data[b]
                    y = label[b]
                    # forward pass
                    for l in range(self.lay_num):
                        output = self.layers[l].forward(x)
                        x = output
                    temp_loss += cross_entropy(output, y)
                    print(temp_loss)
                    # print('train y:', np.argmax(output))
                    # print('test y:', np.argmax(y))
                    if np.argmax(output) == np.argmax(y):
                        temp_acc += 1
                        total_acc += 1
                    # backward pass
                    dy = y
                    for l in range(self.lay_num-1, -1, -1):
                        dout = self.layers[l].backward(dy)
                        dy = dout
                # result
                temp_loss /= batch_size
                loss = np.append(loss, temp_loss)
                batch_acc = float(temp_acc) / float(batch_size)
                training_acc = float(total_acc) / float((batch_index+batch_size)*(e+1))
                auc = np.append(auc, training_acc)
                print('=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:.4f} === BAcc: {4:.4f} === TAcc: {5:.4f} === '.format(e, epoch, batch_index + batch_size, temp_loss, batch_acc, training_acc))
        # dump weights and bias
        print('saving model and weight...')
        obj = []
        for i in range(self.lay_num):
            cache = self.layers[i].extract()
            obj.append(cache)
        with open(weights_file, 'wb') as handle:
            pickle.dump([obj, loss, auc], handle, protocol = pickle.HIGHEST_PROTOCOL)

    def predict(self, data, label, test_size):

       total_acc = 0
       for i in tqdm(range(test_size)):

           x = data[i]
           y = label[i]
           for l in range(self.lay_num):
               output = self.layers[l].forward(x)
               x = output
           if np.argmax(output) == np.argmax(y):
               total_acc += 1
               
       print('=== Test Size:{0:d} === Predict Acc:{1:.4f} ==='.format(test_size, float(total_acc)/float(test_size)))
