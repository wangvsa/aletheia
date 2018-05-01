import random
import math
import numpy as np
from data_reader import read_data
from aid import AdaptiveDetector
from bits import bit_flip
import glob

def get_flip_error(val):
    while True:
        pos = random.randint(0, 19)
        error = bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    return error


'''
Test the recall of the AID
We insert an error in each frame and see if AID can detect it
'''
def test_0_recall(prefix):
    aid = AdaptiveDetector()

    d5 = read_data(prefix, 1)
    d4 = read_data(prefix, 2)
    d3 = read_data(prefix, 3)
    d2 = read_data(prefix, 4)
    d1 = read_data(prefix, 5)

    # start from the 6th frame
    recall = 0
    for it in range(6, 1001):
        d = read_data(prefix, it)

        # insert an error
        x, y = random.randint(0, 479), random.randint(0, 479)
        org = d[x, y]
        truth = False
        if it % 2 == 0:
            truth = True
            d[x, y] = get_flip_error(org)

        hasError = aid.detect(d, d1, d2, d3, d4, d5)
        if hasError and truth:      # true positive
            recall += 1
        if hasError and not truth:  # false positive
            aid.fp += 1

        aid.it += 1

        d[x, y] = org   # restore the correct value before next detection

        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d

        print("it:", it, " recall:", recall, " fp:", aid.fp)

# Perform the detection on clean data to get false positive rate
def test_k_recall(restart_iter, clean_prefix, error_prefix):
    delay = 6
    recall = np.zeros(delay)

    aid = AdaptiveDetector()
    aid.it = 0
    aid.fp = restart_iter / 2

    # first read previous clean data
    d5 = read_data(clean_prefix, restart_iter-5)
    d4 = read_data(clean_prefix, restart_iter-4)
    d3 = read_data(clean_prefix, restart_iter-3)
    d2 = read_data(clean_prefix, restart_iter-2)
    d1 = read_data(clean_prefix, restart_iter-1)

    for k in range(delay):      # detecting after k iterations
        d = read_data(error_prefix, k)
        recall[k] = aid.detect(d, d1, d2, d3, d4, d5)
        aid.it += 1

        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d
    print recall
    return recall


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

test_0_recall("/home/wangchen/Sources/mantevo/TeaLeaf/TeaLeaf_ref/clean/data_")
#test_fp("/home/wangchen/Flash/Sedov/clean/sedov_hdf5_plt_cnt_")


'''
# test k iterations
if __name__ == "__main__":
    recall = np.zeros(6)
    total = 0
    # First find all restarting points
    directory = "/home/wangchen/Flash/Sedov/"
    for filename in glob.iglob(directory+"*plt_cnt_0000"):
        restart_iter = int(filename.split("_")[1])
        error_prefix = filename[0:-4]
        clean_prefix = directory + "clean/sedov_hdf5_plt_cnt_"
        print(restart_iter, error_prefix)
        recall += test_k_recall(restart_iter, clean_prefix, error_prefix)

        total += 1
    print(recall)
'''
