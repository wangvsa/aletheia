import random
import math
import numpy as np
from data_reader import read_data
from aid import AdaptiveDetector
from bits import bit_flip

def get_flip_error(val):
    while True:
        pos = random.randint(0, 20)
        error = bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    return error


'''
Test the recall of the AID
We insert an error in each frame and see if AID can detect it
'''
def test_recall(prefix):
    aid = AdaptiveDetector()

    d5 = read_data(prefix, 0)
    d4 = read_data(prefix, 1)
    d3 = read_data(prefix, 2)
    d2 = read_data(prefix, 3)
    d1 = read_data(prefix, 4)

    # start from the 6th frame
    recall = 0
    for it in range(5, 1001):
        d = read_data(prefix, it)

        # insert an error
        org = d[100, 100]
        truth = False
        if it % 2 == 0:
            truth = True
            d[100, 100] = get_flip_error(org)

        hasError = aid.detect(d, d1, d2, d3, d4, d5)
        if hasError and truth:      # true positive
            recall += 1
        if hasError and not truth:  # false positive
            aid.fp += 1

        aid.it += 1

        d[100, 100] = org   # restore the correct value before next detection

        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d

        print("it:", it, " recall:", recall, " fp:", aid.fp)


# Perform the detection on clean data to get false positive rate
def test_fp(prefix):
    aid = AdaptiveDetector()

    d5 = read_data(prefix, 0)
    d4 = read_data(prefix, 1)
    d3 = read_data(prefix, 2)
    d2 = read_data(prefix, 3)
    d1 = read_data(prefix, 4)

    # start from the 6th frame
    for it in range(5, 1001):
        d = read_data(prefix, it)
        aid.fp += aid.detect(d, d1, d2, d3, d4, d5)
        aid.it += 1
        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d
        print("it:", it, " fp:", aid.fp)

test_recall("/home/wangchen/Flash/Sedov/clean/sedov_hdf5_plt_cnt_")
#test_fp("/home/wangchen/Flash/Sedov/clean/sedov_hdf5_plt_cnt_")
