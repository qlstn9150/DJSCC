import argparse
import os
import math
import numpy as np
from utils import *
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree

class Encoder:
    def quantize(block, component):
        q = load_quantization_table(component)
        return (block / q).round().astype(np.int32)


    def block_to_zigzag(block):
        return np.array([block[point] for point in zigzag_points(*block.shape)])


    def dct_2d(image):
        return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


    def run_length_encode(arr):
    # determine where the sequence is ending prematurely
        last_nonzero = -1
        for i, elem in enumerate(arr):
            if elem != 0:
                last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
        symbols = []

    # values are binary representations of array elements using SIZE bits
        values = []

        run_length = 0

        for i, elem in enumerate(arr):
            if i > last_nonzero:
                symbols.append((0, 0))
                values.append(int_to_binstr(0))
                break
            elif elem == 0 and run_length < 15:
                run_length += 1
            else:
                size = bits_required(elem)
                symbols.append((run_length, size))
                values.append(int_to_binstr(elem))
                run_length = 0
        return symbols, values


    def write_to_file(filepath, dc, ac, blocks_count, tables):
        try:
            f = open(filepath, 'w')
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e

        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # 16 bits for 'table_size'
            f.write(uint_to_binstr(len(tables[table_name]), 16))

            for key, value in tables[table_name].items():
                if table_name in {'dc_y', 'dc_c'}:
                # 4 bits for the 'category'
                # 4 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                    f.write(uint_to_binstr(key, 4))
                    f.write(uint_to_binstr(len(value), 4))
                    f.write(value)
                else:
                # 4 bits for 'run_length'
                # 4 bits for 'size'
                # 8 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                    f.write(uint_to_binstr(key[0], 4))
                    f.write(uint_to_binstr(key[1], 4))
                    f.write(uint_to_binstr(len(value), 8))
                    f.write(value)

    # 32 bits for 'blocks_count'
        f.write(uint_to_binstr(blocks_count, 32))

        for b in range(blocks_count):
            for c in range(3):
                category = bits_required(dc[b, c])
                symbols, values = run_length_encode(ac[b, :, c])

                dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
                ac_table = tables['ac_y'] if c == 0 else tables['ac_c']

                f.write(dc_table[category])
                f.write(int_to_binstr(dc[b, c]))

                for i in range(len(symbols)):
                    f.write(ac_table[tuple(symbols[i])])
                    f.write(values[i])
        f.close()