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
    print f.keys()
    data = f[var_name][:]
    return data

# Stack a shape of (N, 1, ny, nx) to shape (ny, N*nx)
def stack(data):
    data = np.vstack(data)
    data = np.hstack(data)
    return data


def show_heatmap(filename):
    data = hdf5_to_numpy(filename)
    data = stack(data)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()


def test(filename):
    data = hdf5_to_numpy(filename, "dens")
    print "dens:", data.shape, data
    data = stack(data)
    print "dens:", data.shape, data

    data = hdf5_to_numpy(filename, "coordinates")
    print "coordinates:", data.shape, data

    data = hdf5_to_numpy(filename, "processor number")
    print "processor number:", data.shape, data


if __name__ == "__main__":
    if len(sys.argv) != 2 :
        sys.exit("Usage: python flash_plot.py hdf5_plt_file_path")
    path = sys.argv[1]
    show_heatmap(path)
    #test(path)
