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

if __name__ == "__main__":
    dataset = read_hdf5_dataset(sys.argv[1])
    print dataset.shape

# Copy the dataset N times and inject errors
# Return dataset and the weather has errors
def preprocess_for_classifier(dataset, N = 1):
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

            ''' Old method to inject error
            error = random.uniform(0.01*d, 0.9*d)
            sign = 1 if random.randint(0,1) == 1 else -1
            dataset[i][x,y] = error
            '''

            bit_pos = random.randint(1, 10)
            error = bit_flip(dataset[i][x,y], bit_pos)
            if math.isnan(error) or math.isinf(error):
                has_error[i] = 0
                continue
            if error < 0.0001:
                error = 0.0001
            if error > 5*d:
                error = 5*d
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

