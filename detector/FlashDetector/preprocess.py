'''
Read the dataset and inject errors
'''
import numpy as np
import glob
import h5py
import sys
import random
import math
import os
from skimage.util.shape import view_as_windows, view_as_blocks


# Read from a hdf5 file to a numpy array
def hdf5_to_numpy(filename, var_name="dens"):
    f = h5py.File(filename, 'r')
    # Stack a shape of (1, 1, ny, nx) to shape (ny, nx)
    data = f[var_name][0, 0]
    return data

def split_to_windows(frame, rows, cols, overlap):
    step = cols - overlap
    windows = view_as_windows(frame, (rows, cols), step = step)
    return np.vstack(windows)
def split_to_blocks(frame, rows, cols):
    blocks = view_as_blocks(frame, (rows, cols))
    return np.vstack(blocks)

def get_error_data(data_dir, rows, cols, overlap):
    dataset = []
    for filename in glob.iglob(data_dir+"/error_*hdf5_plt_cnt_*"):
    #for filename in glob.iglob(data_dir+"/error_0_hdf5_plt_cnt_*"):
        if 'forced' in filename in filename:
            os.system('rm '+filename)
            continue

        dens = hdf5_to_numpy(filename, 'dens')
        temp = hdf5_to_numpy(filename, 'temp')
        pres = hdf5_to_numpy(filename, 'pres')

        #shape of (N, rows, cols)
        dens_blocks = split_to_windows(dens, rows, cols, overlap)
        temp_blocks = split_to_windows(temp, rows, cols, overlap)
        pres_blocks = split_to_windows(pres, rows, cols, overlap)
        #dens_blocks /= np.std(dens_blocks, axis=0)
        #temp_blocks /= np.std(temp_blocks, axis=0)
        #pres_blocks /= np.std(pres_blocks, axis=0)

        # combile to 3 channels, shape of (N, 3, rows, cols)
        tmp = np.swapaxes( np.array([dens_blocks, temp_blocks, pres_blocks]), 0, 1)
        print filename, dens_blocks.shape, tmp.shape
        dataset.append(tmp)
    dataset = np.vstack(dataset)
    print "dataset shape:", dataset.shape
    output_file = data_dir.split('/')[-2]
    np.save("error_iter_5", dataset)


# Read all hdf5 files in a given directory
# Each frame is splited into blocks with overlap
def get_clean_data(data_dir, rows, cols, overlap):
    dataset = []
    for filename in glob.iglob(data_dir+"/sedov_hdf5_chk_*"):
        dens = hdf5_to_numpy(filename, 'dens')
        temp = hdf5_to_numpy(filename, 'temp')
        pres = hdf5_to_numpy(filename, 'pres')

        dens_blocks = split_to_windows(dens, rows, cols, overlap)  #shape of (N, rows, cols)
        temp_blocks = split_to_windows(temp, rows, cols, overlap)
        pres_blocks = split_to_windows(pres, rows, cols, overlap)
        dens_blocks /= np.std(dens_blocks, axis=0)
        temp_blocks /= np.std(temp_blocks, axis=0)
        pres_blocks /= np.std(pres_blocks, axis=0)

        # combile to 3 channels, shape of (N, 3, rows, cols)
        tmp = np.swapaxes( np.array([dens_blocks, temp_blocks, pres_blocks]), 0, 1)
        print filename
        print dens_blocks.shape, tmp.shape
        dataset.append(tmp)
    dataset = np.vstack(dataset)
    print "dataset shape:", dataset.shape
    output_file = data_dir.split('/')[-2]
    np.save(output_file, dataset)

def test_min_max(data_dir):
    for filename in glob.iglob(data_dir+"/*.npy"):
        data = np.load(filename)
        print filename, data.shape, ', min:', np.unravel_index(np.argmin(data), data.shape), 'max:',  np.unravel_index(np.argmax(data), data.shape)

#get_clean_data(sys.argv[1], 60, 60, 20)
get_error_data(sys.argv[1], 60, 60, 20)
#test_min_max(sys.argv[1])
