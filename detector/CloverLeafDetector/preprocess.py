'''
Read the dataset and inject errors
'''
import numpy as np
import glob
import h5py
import random
from bits import bit_flip
import math


# Read from a hdf5 file to a numpy array
def hdf5_to_numpy(filename):
    f = h5py.File(filename, 'r')
    data = f[list(f.keys())[0]][:]
    return data


# Read all hdf5 files in a given directory
# Each file contains a matrix with shape (rows, cols)
# Split it into many blocks with give shape (m, n)
# Return all blocks a new numpy array with shape (N, m, n)
def read_hdf5_dataset(directory, m, n):
    dataset = []
    for filename in glob.iglob(directory+"/*.h5"):
        frame = hdf5_to_numpy(filename)
        blocks = split_to_blocks(frame, m, n)
        dataset.append(blocks)
    return np.vstack(dataset)

# Split a given 2d array into many equal size blocks
def split_to_blocks(arr, rows, cols):
    h, w = arr.shape
    return (arr.reshape(h//rows, rows, -1, cols)
                .swapaxes(1,2)
                .reshape(-1, rows, cols))


# Copyt the dataset N times and inject errors
# Return dataset and the weather has errors
def preprocess_for_classifier(dataset, N = 10):
    dataset = np.vstack([dataset] * N) #  copy the dataset N times
    has_error = np.zeros(len(dataset))
    sign = 1
    THREASHOLD = 0.5
    for i in range(len(dataset)):
        if random.randint(0, 1):
            x = random.randint(0, dataset[i].shape[0]-1)
            y = random.randint(0, dataset[i].shape[1]-1)
            has_error[i] = 1

            d = dataset[i][x,y]
            '''
            t = random.uniform(d, 5*d)
            sign = 1 if random.randint(0,1) == 1 else -1
            error = dataset[i][x,y] + t
            dataset[i][x,y] = error
            '''

            bit_pos = random.randint(1, 10)
            #bit_pos = 8
            error = bit_flip(dataset[i][x,y], bit_pos)
            if math.isnan(error) or math.isinf(error):
                has_error[i] = 0
                continue
            #if error < 0.0001:
            #    error = 0.0001
            #if error > 5*d:
            #    error = 5*d
            #print 'old:', dataset[i][x,y], ', pos:', bit_pos, ', new:', error
            dataset[i][x,y] = error
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

