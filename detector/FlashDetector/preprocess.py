'''
Read the dataset and inject errors
'''
import numpy as np
import glob
import h5py
import sys
import random
import math
from skimage.util.shape import view_as_windows
from bits import bit_flip


# Read from a hdf5 file to a numpy array
def hdf5_to_numpy(filename, var_name="dens"):
    f = h5py.File(filename, 'r')
    data = f[var_name][:]
    # Stack a shape of (N, 1, ny, nx) to shape (ny, N*nx)
    data = np.hstack(np.vstack(data))
    return data

def split_to_windows(frame, rows, cols, overlap):
    step = cols - overlap
    windows = view_as_windows(frame, (rows, cols), step = step)
    return np.vstack(windows)

# Read all hdf5 files in a given directory
# Each frame is splited into blocks with overlap
def get_classifier_training_data(data_dir, rows, cols, overlap=0, N=1):
    dataset = None
    for filename in glob.iglob(data_dir+"/*hdf5_plt_cnt*"):
        frame = hdf5_to_numpy(filename)
        windows = split_to_windows(frame, rows, cols, overlap)
        if dataset is None :
            dataset = windows
        else :
            dataset = np.vstack((dataset, windows))

    dataset = np.vstack([dataset] * N) #  copy the dataset N times
    has_error = np.zeros(len(dataset))
    THREASHOLD = 0.01
    for i in range(len(dataset)):
        if random.randint(0, 10)==5:
            x = random.randint(0, dataset[i].shape[0]-1)
            y = random.randint(0, dataset[i].shape[1]-1)

            has_error[i] = 1

            d = dataset[i][x,y]
            d_max = np.max(dataset[i])
            d_min = np.min(dataset[i])

            bit_pos = random.randint(0, 15)
            error = bit_flip(dataset[i][x,y], bit_pos)
            if math.isnan(error) or math.isinf(error): # or (abs(error/d-1) < THREASHOLD):
                has_error[i] = 0
                continue

            # Limit error withim [d_min/5, d_max*5]
            error = min(10e+5, error)
            error = max(-10e+5, error)

            #print 'old:', dataset[i][x,y], ', pos:', bit_pos, ', new:', error
            dataset[i][x,y] = error
        std = np.std(dataset[i])
        if std == 0: std = np.max(dataset[i])
        dataset[i] = dataset[i] / std
    return dataset, has_error



def get_classifier_test_data(data_dir, rows, cols, overlap):
    dataset = []
    has_error = []
    for filename in glob.iglob(data_dir+"/*hdf5_plt_cnt*"):
        frame = hdf5_to_numpy(filename)
        blocks = split_to_windows(frame, rows, cols, overlap)
        # Inser an error
        if random.randint(0, 1):
            blockId = random.randint(0, blocks.shape[0]-1)
            x = random.randint(0, blocks.shape[1]-1)
            y = random.randint(0, blocks.shape[2]-1)
            bit_pos = random.randint(0, 15)
            error = bit_flip(blocks[blockId][x,y], bit_pos)
            if math.isnan(error) or math.isinf(error):
                has_error.append(0)
            else:
                error = min(error, 10e+10)
                blocks[blockId][x,y] = error
                has_error.append(1)
        else:
            has_error.append(0)

        std = np.std(blocks)
        if std == 0: std = np.max(blocks)
        blocks = blocks / std
        dataset.append(blocks)

    return dataset, has_error
