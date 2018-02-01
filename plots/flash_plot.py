import sys
import h5py
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read from a hdf5 file to a numpy array
def hdf5_to_numpy(filename, var_name="dens"):
    f = h5py.File(filename, 'r')
    data = f[var_name][:]
    return data

def show_heatmap(filename):
    data = hdf5_to_numpy(filename)[100][0]
    print data.shape
    print data
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) != 2 :
        sys.exit("Usage: python plot dataset_directory")
    path = sys.argv[1]
    show_heatmap(path)
