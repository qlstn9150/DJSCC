import tensorflow as tf
import os
import numpy as np
from keras.models import load_model
from AutoencoderModel import NormalizationNoise
from keras.datasets import cifar10
from keras import backend as K
from skimage.metrics import structural_similarity
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
import h5py

####################################################################################################
# 데이터 준비
(trainX, _), (testX, _) = cifar10.load_data()


def normalize_pixels(train_data, test_data):
    train_norm = train_data.astype('float32')
    test_norm = test_data.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


x_train, x_test = normalize_pixels(trainX, testX)


####################################################################################################
# Basic CNN
def comp_eval(x_test, compression_ratios, snr, mode='multiple'):
    model_dic = {'SNR': [], 'Pred_Images': [], 'PSNR': [], 'SSIM': []}
    model_dic['SNR'].append(snr)
    for comp_ratio in compression_ratios:
        tf.keras.backend.clear_session()
        path = './checkpoints/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        ssim = structural_similarity(testX, pred_images, multichannel=True)
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Pred_Images'].append(pred_images)
        model_dic['PSNR'].append(psnr)
        model_dic['SSIM'].append(ssim)
    return model_dic


# DenseNet
def comp_eval_DN(x_test, compression_ratios, snr, mode='multiple'):
    model_dic = {'SNR': [], 'Pred_Images': [], 'PSNR': [], 'SSIM': []}
    model_dic['SNR'].append(snr)
    for comp_ratio in compression_ratios:
        tf.keras.backend.clear_session()
        path = './checkpoints_DN/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        ssim = structural_similarity(testX, pred_images, multichannel=True)
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Pred_Images'].append(pred_images)
        model_dic['PSNR'].append(psnr)
        model_dic['SSIM'].append(ssim)
    return model_dic


# 그래프 생성
def comp_plot(x_test, compression_ratios, snr_lst, title, x_lablel, y_label):
    markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
    colors = ['#800080', '#FF00FF', '#000080', '#008080', '#00FFFF', '#008000', '#00FF00']
    history = []
    i = 0
    for snr in snr_lst:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = comp_eval(x_test, compression_ratios, snr, mode='multiple')
        history.append(model_dic)
        label = 'DJSCC (SNR={0}dB)'.format(snr)
        plt.plot(compression_ratios, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1

    for snr in snr_lst:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = comp_eval_DN(x_test, compression_ratios, snr, mode='multiple')
        history.append(model_dic)
        label = 'DenseNet (SNR={0}dB)'.format(snr)
        plt.plot(compression_ratios, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1
        plt.title(title)
        plt.xlabel(x_lablel)
        plt.ylabel(y_label)
        plt.grid(True)

    plt.ylim(10, 20)
    plt.legend(loc='lower right')
    plt.show()
    return history


# 실행
compression_ratios = [0.06, 0.26, 0.49]
snr_train = [0, 10, 20]

history = comp_plot(x_test, compression_ratios, snr_train, title='AWGN Channel', x_lablel='k/n', y_label='PSNR (dB)')


####################################################################################################
# Basic CNN
def snr_eval(x_test, comp_ratio, snr_test, snr_train):
    model_dic = {'Train_snr': [snr_train], 'Test_snr': [], 'PSNR': []}
    for snr in snr_test:
        tf.keras.backend.clear_session()
        path = './checkpoints/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr_train)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Test_snr'].append(snr)
        model_dic['PSNR'].append(psnr)
    return model_dic


# DenseNet
def snr_eval_DN(x_test, comp_ratio, snr_test, snr_train):
    model_dic = {'Train_snr': [snr_train], 'Test_snr': [], 'PSNR': []}
    for snr in snr_test:
        tf.keras.backend.clear_session()
        path = './checkpoints_DN/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr_train)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Test_snr'].append(snr)
        model_dic['PSNR'].append(psnr)
    return model_dic


# 그래프 생성
def snr_plot(x_test, compression_ratio, snr_train, title, x_lablel, y_label):
    markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
    colors = ['#800080', '#FF00FF', '#000080', '#008080', '#00FFFF', '#008000', '#00FF00']
    history = []
    i = 0
    for snr in snr_train:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = snr_eval(x_test, compression_ratio, snr_test, snr)
        history.append(model_dic)
        label = 'DJSCC (SNR_train={0}dB)'.format(snr)
        plt.plot(snr_test, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1

    for snr in snr_train:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = snr_eval_DN(x_test, compression_ratio, snr_test, snr)
        history.append(model_dic)
        label = 'DenseNet (SNR_train={0}dB)'.format(snr)
        plt.plot(snr_test, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1
        plt.title(title)
        plt.xlabel(x_lablel)
        plt.ylabel(y_label)
        plt.grid(True)

    plt.ylim(10, 20)
    plt.legend(loc='lower right')
    plt.show()
    return history


# 실행
snr_train = [0, 10, 20]
snr_test = [2, 4, 7, 10, 13, 16, 18, 22, 25, 27]
comp_ratio = 0.06
history2 = snr_plot(x_test, comp_ratio, snr_train, title='AWGN Channel', x_lablel='SNR_test (dB)', y_label='PSNR (dB)')

####################################################################################################
'''
j=6          
test_img=np.array([x_test[j]])
true_img=testX[j]
comp_ratio=0.17
epoch_range=550
epoch_step=50
snr=10
preds = {'SNR':[], 'PSNR':[], 'SSIM':[], 'Epoch':[], 'Image':[]}
preds['SNR'].append(snr)
preds['PSNR'].append('PSNR')
preds['SSIM'].append('SSIM')
preds['Epoch'].append('Epoch')
preds['Image'].append(true_img)
for epoch in range(50,epoch_range,epoch_step):
    tf.keras.backend.clear_session()
    path = './CKPT_ByEpochs/CompRatio_{0}SNR{1}/Autoencoder_Epoch_{2}.h5'.format(comp_ratio, snr, epoch)
    autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
    K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
    pred_images = autoencoder.predict(test_img)[0]*255
    pred_images = pred_images.astype('uint8')
    preds['SNR'].append(snr)
    preds['PSNR'].append(peak_signal_noise_ratio(true_img, pred_images))
    preds['SSIM'].append(structural_similarity(true_img, pred_images, multichannel=True))
    preds['Epoch'].append(epoch)
    preds['Image'].append(pred_images)
snr_lst = [10,20]
col=epoch_range//epoch_step+1 
rows=len(snr_lst)    
i=0 
plt_step=2
for j in range((len(preds['Image'])+2)//plt_step):
    # define subplot
    plt.subplot(rows,col,i+1)
    # plot raw pixel data
    print(preds['Epoch'][i])
    fig = plt.imshow(preds['Image'][i])
    if i%col==0:
        plt.title(str(preds['PSNR'][i])+'/'+str(preds['SSIM'][i]))
        plt.ylabel('SNR {0} dB'.format(preds['SNR'][i]))
        plt.show()
    else:
        plt.title(str(int(preds['PSNR'][i]))+'/{0:.3f}'.format(preds['SSIM'][i]))
        plt.show()
    i=i+plt_step        
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    '''