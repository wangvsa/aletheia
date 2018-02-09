'''
Read the dataset and inject errors
'''
import numpy as np
import glob
import h5py
import sys
import random
from bits import bit_flip
import math


# Read from a hdf5 file to a numpy array
def hdf5_to_numpy(filename, var_name="dens"):
    f = h5py.File(filename, 'r')
    data = f[var_name][:]
    shape = data.shape
    data = np.reshape(data, (shape[0], shape[2], shape[3]))
    return data

# Read all hdf5 files in a given directory
def read_hdf5_dataset(directory):
    dataset = None
    for filename in glob.iglob(directory+"/*hdf5_plt_cnt*"):
        if dataset is None :
            dataset = hdf5_to_numpy(filename)
        else :
            blocks = hdf5_to_numpy(filename)
            dataset = np.vstack((dataset, blocks))
    return dataset

def get_classifier_test_data(data_dir):
    dataset = []
    has_error = []
    for filename in glob.iglob(data_dir+"/*hdf5_plt_cnt*"):

        blocks= hdf5_to_numpy(filename)

        # Inser an error
        if random.randint(0, 1):
            blockId = random.randint(0, blocks.shape[0]-1)
            x = random.randint(0, blocks.shape[1]-1)
            y = random.randint(0, blocks.shape[2]-1)
            bit_pos = random.randint(0, 10)
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

# Copy the dataset N times and inject errors
# Return dataset and the weather has errors
def preprocess_for_classifier(dataset, N = 1):
    dataset = np.vstack([dataset] * N) #  copy the dataset N times
    has_error = np.zeros(len(dataset))
    sign = 1
    THREASHOLD = 0.01
    for i in range(len(dataset)):
        if random.randint(0, 10)==5:
            x = random.randint(0, dataset[i].shape[0]-1)
            y = random.randint(0, dataset[i].shape[1]-1)

            has_error[i] = 1

            d = dataset[i][x,y]
            d_max = np.max(dataset[i])
            d_min = np.min(dataset[i])

            bit_pos = random.randint(0, 10)
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

# Copy the dataset N times and inject errors
# Return the dataset and error positions
def preprocess_for_detector(dataset, N = 10):
    dataset = np.vstack([dataset] * N) #  copy the dataset N times
    error_positions = np.zeros((len(dataset), 2))
    sign = 1
    THREASHOLD = 0.5
    for i in range(len(dataset)):
        x = np.random.randint(0, dataset[i].shape[0])
        y = np.random.randint(0, dataset[i].shape[1])
        error_positions[i] = [x, y]
        error = np.random.uniform(dataset[i][x, y] * THREASHOLD, 2.5, size=1)  # from origin_val*THREASHOLD ~ 1.0
        dataset[i][x, y] = dataset[i][x,y] + sign * error
        #dataset[i][x, y] = min(dataset[i][x,y], 2.5)
        #dataset[i][x, y] = max(dataset[i][x,y], 0)
        sign = sign * -1
        #dataset[i][x, y] = 1
    return dataset, error_positions
