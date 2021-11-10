# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:31:29 2019

@author: Danish
Wrapper File for 1. Compute pwr combined (Real, Imag), (Extract R & I parts) Generate single distribution, Separate Sending, (R&I)
"""

from keras.layers import Conv2D, Layer, Input, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers import Activation, BatchNormalization, Cropping2D
from keras.layers import Concatenate, Lambda, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
#from keras.optimizers import adam_v2
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import keras
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import os 

class NormalizationNoise(Layer):
    def __init__(self, snr_db_def = 20, P_def=1, name='NormalizationNoise', **kwargs):
        self.snr_db = K.variable(snr_db_def, name='SNR_db')
        self.P = K.variable(P_def, name='Power')
        self._name = name
        super(NormalizationNoise, self).__init__(**kwargs)
    def call(self, z_tilta): 
        with tf.name_scope('Normalization_Layer'):
            z_tilta = tf.dtypes.cast(z_tilta, dtype='complex128', name='ComplexCasting')+1j
            lst = z_tilta.get_shape().as_list() 
            lst.pop(0) 
            k = np.prod(lst, dtype='float32')
            z_conjugateT = tf.math.conj(tf.transpose(z_tilta, perm=[0,2,1,3], name='transpose'), name='z_ConjugateTrans')
            sqrt1 = tf.dtypes.cast(tf.math.sqrt(k*self.P, name='NormSqrt1'), dtype='complex128',name='ComplexCastingNorm')
            sqrt2 = tf.math.sqrt(z_conjugateT*z_tilta, name='NormSqrt2')
            div = tf.math.divide(z_tilta,sqrt2, name='NormDivision')
            z = tf.math.multiply(sqrt1,div, name='Z')   
            
        with tf.name_scope('PowerConstraint'):     
            ############# Implementing Power Constraint ##############
            z_star = tf.math.conj(tf.transpose(z, perm=[0,2,1,3], name='transpose_Pwr'), name='z_star')
            prod = z_star*z 
            real_prod = tf.dtypes.cast(prod , dtype='float32', name='RealCastingPwr')
            pwr = tf.math.reduce_mean(real_prod)
            cmplx_pwr = tf.dtypes.cast(pwr, dtype='complex128', name='PowerComplexCasting')+1j
            pwr_constant = tf.constant(1.0, name ='PowerConstant')
            Z = tf.cond(pwr>pwr_constant, lambda: tf.math.divide(z,cmplx_pwr), lambda: z, name='Z_fixed')
            
        with tf.name_scope('AWGN_Layer'):     
            k=k.astype('float64')
            snr = 10**(self.snr_db/10.0)
            snr = tf.dtypes.cast(snr, dtype='float64', name='Float32_64Cast')
            ########### Calculating signal power ########### 
            abs_val = tf.math.abs(Z, name='abs_val')
            summation = tf.math.reduce_sum(tf.math.square(abs_val, name='sq_awgn'), name='Summation')
            sig_pwr = tf.math.divide(summation,k, name='Signal_Pwr')
            noise_pwr = tf.math.divide(sig_pwr,snr, name='Noise_Pwr')
            noise_sigma = tf.math.sqrt(noise_pwr/2, name='Noise_Sigma')

            z_img = tf.math.imag(Z, name = 'Z_imag')
            z_real = tf.math.real(Z, name = 'Z_real')
            rand_dist = tf.random.normal(tf.shape(z_real), dtype=tf.dtypes.float64, name='RandNormalDist')
            noise = tf.math.multiply(noise_sigma, rand_dist, name='Noise')
            z_cap_Imag = tf.math.add(z_img, noise, name='z_cap_Imag') 
            z_cap_Imag = tf.dtypes.cast(z_cap_Imag, dtype='float32', name='NoisySignal_Imag')       
            z_cap_Real = tf.math.add(z_real, noise, name='z_cap_Real') 
            z_cap_Real = tf.dtypes.cast(z_cap_Real, dtype='float32', name='NoisySignal_Real')
            
            return z_cap_Real  

class ModelCheckponitsHandler(tf.keras.callbacks.Callback):
  def __init__(self, comp_ratio, snr_db, autoencoder, step):
    super(ModelCheckponitsHandler, self).__init__()
    self.comp_ratio = comp_ratio
    self.snr_db = snr_db
    self.step = step
    self.autoencoder = autoencoder
  def on_epoch_begin(self, epoch, logs=None):
    if epoch%self.step==0:
        os.makedirs('./CKPT_ByEpochs_DN/CompRatio_'+str(self.comp_ratio)+'SNR'+str(self.snr_db), exist_ok=True)
        path = './CKPT_ByEpochs_DN/CompRatio_'+str(self.comp_ratio)+'SNR'+str(self.snr_db)+'/Autoencoder_Epoch_'+str(epoch)+'.h5'
        self.autoencoder.save(path)
        print('\nModel Saved After {0} epochs.'.format(epoch))


def Calculate_filters(comp_ratio, F=5, n=3072):
    K = (comp_ratio*n)/F**2
    return int(K)

def CNN_AE(x_train, x_test, nb_epoch, comp_ratio, batch_size, c, snr, saver_step=50):
    ############################### Buliding Encoder ##############################
    input_images = Input(shape=(32,32,3))
    #1st convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal')(input_images)
    prelu1 = PReLU()(conv1)
    #2nd convolutional layer
    conv2 = Conv2D(filters=80, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal')(prelu1)
    prelu2 = PReLU()(conv2)
    #3rd convolutional layer
    conv3 = Conv2D(filters=50, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu2)
    prelu3 = PReLU()(conv3)
    #4th convolutional layer
    conv4 = Conv2D(filters=40, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu3)
    prelu4 = PReLU()(conv4)
    #5th convolutional layer
    conv5 = Conv2D(filters=c, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu4)
    encoder = PReLU()(conv5)
    
    real_prod = NormalizationNoise()(encoder)
    
    ############################### Building Decoder ##############################
    #1st Deconvolutional layer
    decoder = Conv2DTranspose(filters=40, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(real_prod)
    decoder = PReLU()(decoder)
    #2nd Deconvolutional layer
    decoder = Conv2DTranspose(filters=50, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    #3rd Deconvolutional layer
    decoder = Conv2DTranspose(filters=80, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    #4th Deconvolutional layer
    decoder = Conv2DTranspose(filters=16, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    #decoder_up = UpSampling2D((2,2))(decoder)
    #5th Deconvolutional layer
    decoder = Conv2DTranspose(filters=3, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal', activation ='sigmoid')(decoder)
    #decoder = PReLU()(decoder)
    decoder_up = UpSampling2D((2,2))(decoder)
    decoder = Cropping2D(cropping=((13,13),(13,13)))(decoder_up)
    
    ############################### Buliding Models ###############################
    autoencoder = Model(input_images, decoder)
    
    
    K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
    autoencoder.compile(optimizer= Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    autoencoder.summary()
    print('\t-----------------------------------------------------------------')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t| Training Parameters: Filter Size: {0}, Compression ratio: {1} |'.format(c, comp_ratio))
    print('\t|\t\t\t  SNR: {0} dB\t\t\t\t|'.format(snr))
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t-----------------------------------------------------------------')
    tb = keras.callbacks.tensorboard_v1.TensorBoard(
        log_dir='./Tensorboard/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)))
    os.makedirs('./checkpoints/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)), exist_ok=True)
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(
        filepath='./checkpoints/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr))+'/Autoencoder.h5', monitor='val_loss', save_best_only=True)
    ckpt = ModelCheckponitsHandler(comp_ratio, snr, autoencoder, step=saver_step)
    history = autoencoder.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,  callbacks=[tb, checkpoint, ckpt], validation_data=(x_test,x_test))
    return history



layers_in_block = {'DenseNet-121' : [6, 12, 24, 16],
                   'DenseNet-169' : [6, 12, 32, 32],
                   'DenseNet-201' : [6, 12, 48, 32],
                   'DenseNet-265' : [6, 12, 64, 48]}

def Conv_Block(x, growth_rate, activation='relu'):
    '''
    x_l = BatchNormalization()(x)
    x_l = Activation(activation)(x_l)
    x_l = Conv2D(growth_rate*4, (1, 1), padding='same', kernel_initializer='he_normal')(x_l)
    x_l = Dropout(rate=0.5)(x_l)
    '''
    x_l = BatchNormalization()(x)
    x_l = Activation(activation)(x_l)
    x_l = Conv2D(growth_rate, (1, 1), padding='same', kernel_initializer='he_normal')(x_l)
    x_l = Dropout(rate=0.5)(x_l)
    x = Concatenate()([x, x_l])
    return x

def Dense_Block(x, layers, growth_rate=32):
    for i in range(layers):
        x = Conv_Block(x, growth_rate)
    return x

def Transition_Layer(x, compression_factor=0.5, activation='relu'):
    reduced_filters = int(K.int_shape(x)[-1] * compression_factor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(reduced_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Dropout(rate=0.5)(x)
    x = AveragePooling2D((2, 2), padding='same', strides=2)(x)
    return x

def de_Conv_Block(x, growth_rate, activation='relu'):
    #x_l = Concatenate()([x, x_l]) #분리해야함
    #x_l = AveragePooling2D((2, 2), padding='same', strides=2)(x)
    x_l = Lambda(lambda x: tf.split(x, num_or_size_splits=1, axis=1))(x)
    x_l = Conv2DTranspose(growth_rate, (1, 1), padding='same', kernel_initializer='he_normal')(x)
    x_l = Activation(activation)(x_l)
    x_l = BatchNormalization()(x_l)
    '''
    x_l = Conv2DTranspose(growth_rate*4, (1, 1), padding='same', kernel_initializer='he_normal')(x_l)
    x_l = Activation(activation)(x_l)
    x = BatchNormalization()(x_l)
    '''
    return x_l

def de_Dense_Block(x, layers, growth_rate=32):
    for i in range(layers):
        x = de_Conv_Block(x, growth_rate)
    return x

def de_Transition_Layer(x, compression_factor=0.5, activation='relu'):    
    reduced_filters = int(K.int_shape(x)[-1] * compression_factor)
    x = UpSampling2D((2, 2))(x)    
    x = Conv2DTranspose(reduced_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    return x

def DenseNet_AE(x_train, x_test, nb_epoch, comp_ratio, batch_size, c, snr, saver_step=50):
    ############################### Buliding Encoder ##############################
    model_input = Input( shape= (32, 32, 3), dtype='float32' )
    densenet_type='DenseNet-121'
    base_growth_rate = 32

    x = Conv2D(base_growth_rate*2, (7, 7), padding='same', strides=2, kernel_initializer='he_normal')(model_input) # (224, 224, 3) -> (112, 112, 64)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), padding='same', strides=2)(x) # (112, 112, 64) -> (56, 56, 64)

    #Dense_Block 4개, Transition_Layer 3개    
    x = Dense_Block(x, layers_in_block[densenet_type][0], base_growth_rate)
    x = Transition_Layer(x, compression_factor=0.5)
    x = Dense_Block(x, layers_in_block[densenet_type][1], base_growth_rate)
    x = Transition_Layer(x, compression_factor=0.5)
    x = Dense_Block(x, layers_in_block[densenet_type][2], base_growth_rate)
    x = Transition_Layer(x, compression_factor=0.5)
    encoder = Dense_Block(x, layers_in_block[densenet_type][3], base_growth_rate)

    ############################### Noise Channel ##############################
    real_prod   = NormalizationNoise()(encoder)


    ############################### Buliding Decoder ##############################
    #Dense_Block 4개, Transition_Layer 3개  
    x = de_Dense_Block(real_prod, layers_in_block[densenet_type][3], base_growth_rate)
    x = de_Transition_Layer(x)
    x = de_Dense_Block(x, layers_in_block[densenet_type][2], base_growth_rate)
    x = de_Transition_Layer(x)
    x = de_Dense_Block(x, layers_in_block[densenet_type][1], base_growth_rate)
    x = de_Transition_Layer(x)
    x = de_Dense_Block(x, layers_in_block[densenet_type][0], base_growth_rate)

    #output_layer
    x = Activation('relu')(x)
    decoder = BatchNormalization()(x)
    decoder = UpSampling2D((2,2))(decoder)
    decoder = Conv2DTranspose(3, (7, 7), padding='same', strides=2, kernel_initializer='he_normal', activation ='sigmoid')(decoder)
    
    ############################### Buliding Models ###############################
    autoencoder = Model(model_input, decoder)
    
    
    K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
    autoencoder.compile(optimizer= Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    #autoencoder.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), loss='mse', metrics=['accuracy'])
    autoencoder.summary()
    print('\t-----------------------------------------------------------------')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t| Training Parameters: Filter Size: {0}, Compression ratio: {1} |'.format(c, comp_ratio))
    print('\t|\t\t\t  SNR: {0} dB\t\t\t\t|'.format(snr))
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t-----------------------------------------------------------------')
    tb = keras.callbacks.tensorboard_v1.TensorBoard(
        log_dir='./Tensorboard_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)))
    os.makedirs('./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)), exist_ok=True)
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(
        filepath='./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)) + '/Autoencoder.h5',
        monitor='val_loss', save_best_only=True)
    ckpt = ModelCheckponitsHandler(comp_ratio, snr, autoencoder, step=saver_step)
    history = autoencoder.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
                              callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))
    return history


##############################################
def DenseNet1_AE(x_train, x_test, nb_epoch, comp_ratio, batch_size, c, snr, saver_step=50):
    ############################### Buliding Encoder ##############################
    ''' Correspondance of different arguments w.r.t to literature: filters = K, kernel_size = FxF, strides = S'''
    input_images = Input(shape=(32, 32, 3))
    # P = Input(shape=(), name='Power')
    # snr_db = Input(shape=(), name='SNR_DB')
    # 1st convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(5, 5), strides=2, padding='valid', kernel_initializer='he_normal')(
        input_images)
    prelu1 = PReLU()(conv1)
    # 2nd convolutional layer
    conv2 = Conv2D(filters=80, kernel_size=(5, 5), strides=2, padding='valid', kernel_initializer='he_normal')(prelu1)
    prelu2 = PReLU()(conv2)
    # 3rd convolutional layer
    conv3 = Conv2D(filters=50, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal')(prelu2)
    prelu3 = PReLU()(conv3)
    # 4th convolutional layer
    conv4 = Conv2D(filters=50, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal')(prelu3)
    prelu4 = PReLU()(conv4)

    prelu4 = Concatenate()([prelu3, prelu4])  # DenseNet
    # prelu4 = Add()([prelu3, prelu4])  # ResNet

    # 5th convolutional layer
    conv5 = Conv2D(filters=c, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal')(prelu4)
    encoder = PReLU()(conv5)

    real_prod = NormalizationNoise()(encoder)

    ############################### Building Decoder ##############################
    # 1st Deconvolutional layer
    decoder = Conv2DTranspose(filters=50, kernel_size=(5, 5), strides=1, padding='same',
                              kernel_initializer='he_normal')(real_prod)
    decoder = PReLU()(decoder)
    # 2nd Deconvolutional layer
    decoder = Conv2DTranspose(filters=50, kernel_size=(5, 5), strides=1, padding='same',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # 3rd Deconvolutional layer
    decoder = Conv2DTranspose(filters=80, kernel_size=(5, 5), strides=1, padding='same',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # 4th Deconvolutional layer
    decoder = Conv2DTranspose(filters=16, kernel_size=(5, 5), strides=2, padding='valid',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # decoder_up = UpSampling2D((2,2))(decoder)
    # 5th Deconvolutional layer
    decoder = Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=2, padding='valid', kernel_initializer='he_normal',
                              activation='sigmoid')(decoder)
    # decoder = PReLU()(decoder)
    decoder_up = UpSampling2D((2, 2))(decoder)
    decoder = Cropping2D(cropping=((13, 13), (13, 13)))(decoder_up)
    ############################### Buliding Models ###############################
    autoencoder = Model(input_images, decoder)

    K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    autoencoder.summary()
    print('\t-----------------------------------------------------------------')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t| Training Parameters: Filter Size: {0}, Compression ratio: {1} |'.format(c, comp_ratio))
    print('\t|\t\t\t  SNR: {0} dB\t\t\t\t|'.format(snr))
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t-----------------------------------------------------------------')
    tb = keras.callbacks.tensorboard_v1.TensorBoard(
        log_dir='./Tensorboard_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)))
    os.makedirs('./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)), exist_ok=True)
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(
        filepath='./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)) + '/Autoencoder.h5',
        monitor='val_loss', save_best_only=True)
    ckpt = ModelCheckponitsHandler(comp_ratio, snr, autoencoder, step=saver_step)
    history = autoencoder.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
                              callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))
    return history


def DenseNet2_AE(x_train, x_test, nb_epoch, comp_ratio, batch_size, c, snr, saver_step=50):
    ############################### Buliding Encoder ##############################
    ''' Correspondance of different arguments w.r.t to literature: filters = K, kernel_size = FxF, strides = S'''
    input_images = Input(shape=(32, 32, 3))
    # P = Input(shape=(), name='Power')
    # snr_db = Input(shape=(), name='SNR_DB')
    # 1st convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(5, 5), strides=2, padding='valid', kernel_initializer='he_normal')(input_images)
    conv1 = PReLU()(conv1)
    # 2nd convolutional layer
    conv2 = Conv2D(filters=40, kernel_size=(5, 5), strides=2, padding='valid', kernel_initializer='he_normal')(conv1)
    conv2 = PReLU()(conv2)

    # 3rd convolutional layer
    conv3 = Conv2D(filters=40, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = PReLU()(conv3)

    conv3 = Concatenate()([conv2, conv3])  # DenseNet


    # 4th convolutional layer
    conv4 = Conv2D(filters=40, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = PReLU()(conv4)

    conv4 = Concatenate()([conv2, conv3, conv4])  # DenseNet

    # 5th convolutional layer
    conv5 = Conv2D(filters=c, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = PReLU()(conv5)

    real_prod = NormalizationNoise()(conv5)

    ############################### Building Decoder ##############################
    # 1st Deconvolutional layer
    convT1 = Conv2DTranspose(filters=40, kernel_size=(5, 5), strides=1, padding='same',
                              kernel_initializer='he_normal')(real_prod)
    convT1 = PReLU()(convT1)
    # 2nd Deconvolutional layer
    convT2 = Conv2DTranspose(filters=40, kernel_size=(5, 5), strides=1, padding='same',
                              kernel_initializer='he_normal')(convT1)
    convT2 = PReLU()(convT2)

    convT2 = Concatenate()([convT1, convT2])

    # 3rd Deconvolutional layer
    convT3 = Conv2DTranspose(filters=40, kernel_size=(5, 5), strides=1, padding='same',
                              kernel_initializer='he_normal')(convT2)
    convT3 = PReLU()(convT3)

    convT3 = Concatenate()([convT1, convT2, convT3])

    # 4th Deconvolutional layer
    convT4 = Conv2DTranspose(filters=16, kernel_size=(5, 5), strides=2, padding='valid',
                              kernel_initializer='he_normal')(convT3)
    convT4 = PReLU()(convT4)
    # convT4 = UpSampling2D((2,2))(convT4)
    # 5th Deconvolutional layer
    convT5 = Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=2, padding='valid', kernel_initializer='he_normal',
                              activation='sigmoid')(convT4)
    # convT5 = PReLU()(convT5)
    convT5 = UpSampling2D((2, 2))(convT5)
    convT5 = Cropping2D(cropping=((13, 13), (13, 13)))(convT5)
    ############################### Buliding Models ###############################
    autoencoder = Model(input_images, convT5)

    K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    autoencoder.summary()
    print('\t-----------------------------------------------------------------')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t| Training Parameters: Filter Size: {0}, Compression ratio: {1} |'.format(c, comp_ratio))
    print('\t|\t\t\t  SNR: {0} dB\t\t\t\t|'.format(snr))
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t-----------------------------------------------------------------')
    tb = keras.callbacks.tensorboard_v1.TensorBoard(
        log_dir='./Tensorboard_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)))
    os.makedirs('./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)), exist_ok=True)
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(
        filepath='./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)) + '/Autoencoder.h5',
        monitor='val_loss', save_best_only=True)
    ckpt = ModelCheckponitsHandler(comp_ratio, snr, autoencoder, step=saver_step)
    history = autoencoder.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
                              callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))
    return history