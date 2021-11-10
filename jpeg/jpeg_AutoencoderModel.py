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
from keras.models import Model
import keras
import tensorflow as tf

import numpy as np
import tensorflow.keras.backend as K
import os

import argparse
import math
from utils import *
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree
from encoder import Encoder
from decoder import Decoder, JPEGFileReader


class NormalizationNoise(Layer):
    def __init__(self, snr_db_def=20, P_def=1, name='NormalizationNoise', **kwargs):
        self.snr_db = K.variable(snr_db_def, name='SNR_db')
        self.P = K.variable(P_def, name='Power')
        self._name = name
        super(NormalizationNoise, self).__init__(**kwargs)

    def call(self, z_tilta):
        with tf.name_scope('Normalization_Layer'):
            z_tilta = tf.dtypes.cast(z_tilta, dtype='complex128', name='ComplexCasting') + 1j
            lst = z_tilta.get_shape().as_list()
            lst.pop(0)
            k = np.prod(lst, dtype='float32')
            z_conjugateT = tf.math.conj(tf.transpose(z_tilta, perm=[0, 2, 1, 3], name='transpose'),
                                        name='z_ConjugateTrans')
            sqrt1 = tf.dtypes.cast(tf.math.sqrt(k * self.P, name='NormSqrt1'), dtype='complex128',
                                   name='ComplexCastingNorm')
            sqrt2 = tf.math.sqrt(z_conjugateT * z_tilta, name='NormSqrt2')
            div = tf.math.divide(z_tilta, sqrt2, name='NormDivision')
            z = tf.math.multiply(sqrt1, div, name='Z')

        with tf.name_scope('PowerConstraint'):
            ############# Implementing Power Constraint ##############
            z_star = tf.math.conj(tf.transpose(z, perm=[0, 2, 1, 3], name='transpose_Pwr'), name='z_star')
            prod = z_star * z
            real_prod = tf.dtypes.cast(prod, dtype='float32', name='RealCastingPwr')
            pwr = tf.math.reduce_mean(real_prod)
            cmplx_pwr = tf.dtypes.cast(pwr, dtype='complex128', name='PowerComplexCasting') + 1j
            pwr_constant = tf.constant(1.0, name='PowerConstant')
            Z = tf.cond(pwr > pwr_constant, lambda: tf.math.divide(z, cmplx_pwr), lambda: z, name='Z_fixed')

        with tf.name_scope('AWGN_Layer'):
            k = k.astype('float64')
            snr = 10 ** (self.snr_db / 10.0)
            snr = tf.dtypes.cast(snr, dtype='float64', name='Float32_64Cast')
            ########### Calculating signal power ###########
            abs_val = tf.math.abs(Z, name='abs_val')
            summation = tf.math.reduce_sum(tf.math.square(abs_val, name='sq_awgn'), name='Summation')
            sig_pwr = tf.math.divide(summation, k, name='Signal_Pwr')
            noise_pwr = tf.math.divide(sig_pwr, snr, name='Noise_Pwr')
            noise_sigma = tf.math.sqrt(noise_pwr / 2, name='Noise_Sigma')

            z_img = tf.math.imag(Z, name='Z_imag')
            z_real = tf.math.real(Z, name='Z_real')
            rand_dist = tf.random.normal(tf.shape(z_real), dtype=tf.dtypes.float64, name='RandNormalDist')
            noise = tf.math.multiply(noise_sigma, rand_dist, name='Noise')
            z_cap_Imag = tf.math.add(z_img, noise, name='z_cap_Imag')
            z_cap_Imag = tf.dtypes.cast(z_cap_Imag, dtype='float32', name='NoisySignal_Imag')
            z_cap_Real = tf.math.add(z_real, noise, name='z_cap_Real')
            z_cap_Real = tf.dtypes.cast(z_cap_Real, dtype='float32', name='NoisySignal_Real')

            return z_cap_Real


def jpeg(x_train, x_test, nb_epoch, comp_ratio, batch_size, c, snr, saver_step=50):
    ############################### Buliding Encoder ##############################
    input_file = './encoder_input/airplane1.png'
    output_file = './encoder_output/airplane1'

    
    image = Image.open(input_file)
    ycbcr = image.convert('YCbCr')

    npmat = np.array(ycbcr, dtype=np.uint8)

    rows, cols = npmat.shape[0], npmat.shape[1]

    # block size: 8x8
    if rows % 8 == cols % 8 == 0:
        blocks_count = rows // 8 * cols // 8
    else:
        raise ValueError(("the width and height of the image "
                          "should both be mutiples of 8"))

    # dc is the top-left cell of the block, ac are all the other cells
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                block_index += 1
            except NameError:
                block_index = 0

            for k in range(3):
                # split 8x8 block and center the data range on zero
                # [0, 255] --> [-128, 127]
                block = npmat[i:i+8, j:j+8, k] - 128

                dct_matrix = dct_2d(block)
                quant_matrix = quantize(dct_matrix,
                                        'lum' if k == 0 else 'chrom')
                zz = block_to_zigzag(quant_matrix)

                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]

    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(
            flatten(run_length_encode(ac[i, :, 0])[0]
                    for i in range(blocks_count)))
    H_AC_C = HuffmanTree(
            flatten(run_length_encode(ac[i, :, j])[0]
                    for i in range(blocks_count) for j in [1, 2]))

    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}

    write_to_file(output_file, dc, ac, blocks_count, tables)

    ###잡음 채널###
    real_prod = NormalizationNoise()(tables)

    ############################### Building Decoder ##############################
    encoded_image_path = '경로'

    dc, ac, tables, blocks_count = read_image_file(output_file)

    # assuming that the block is a 8x8 square
    block_side = 8

    # assuming that the image height and width are equal
    image_side = int(math.sqrt(blocks_count)) * block_side

    blocks_per_line = image_side // block_side

    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)

    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * block_side
        j = block_index % blocks_per_line * block_side

        for c in range(3):
            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
            quant_matrix = zigzag_to_block(zigzag)
            dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom')
            block = idct_2d(dct_matrix)
            npmat[i:i + 8, j:j + 8, c] = block + 128

    image = Image.fromarray(npmat, 'YCbCr')
    image = image.convert('RGB')
    image.show()
    ###폴더에 이미지 저장해야함

    print('\t-----------------------------------------------------------------')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t| Training Parameters: Filter Size: {0}, Compression ratio: {1} |'.format(c, comp_ratio))
    print('\t|\t\t\t  SNR: {0} dB\t\t\t\t|'.format(snr))
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t---------------------JPEG FINISH---------------------------------')
