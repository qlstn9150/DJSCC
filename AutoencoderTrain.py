# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:39:40 2019

@author: Danish
1. Compute pwr combined (Real, Imag), (Extract R & I parts) Generate single distribution, Separate Sending, (R&I)
"""

from keras.datasets import cifar10
from AutoencoderModel import CNN_AE,DenseNet_AE, DenseNet1_AE, DenseNet2_AE, Calculate_filters
import tensorflow as tf

(trainX, _), (testX, _) = cifar10.load_data()
def normalize_pixels(train_data, test_data):
    train_norm = train_data.astype('float32')
    test_norm = test_data.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0 
    return train_norm, test_norm
x_train, x_test = normalize_pixels(trainX, testX)


SNRs = [0, 10, 20]
compression_ratios = [0.06, 0.26, 0.49]

for snr in SNRs:
    for comp_ratio in compression_ratios:
        tf.keras.backend.clear_session()
        c = Calculate_filters(comp_ratio)
        print('---> System Will Train, Compression Ratio: '+str(comp_ratio)+'. <---')
        #_ = CNN_AE(x_train, x_test, nb_epoch=20, comp_ratio=comp_ratio, batch_size=100, c=c, snr=snr, saver_step=2)
        #_ = DenseNet_AE(x_train, x_test, nb_epoch=20, comp_ratio=comp_ratio, batch_size=100, c=c, snr=snr, saver_step=2)
        _ = DenseNet2_AE(x_train, x_test, nb_epoch=20, comp_ratio=comp_ratio, batch_size=100, c=c, snr=snr, saver_step=2)