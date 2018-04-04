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
    data = f[var_name][0, 0]
    return data

def show_heatmap(filename, npy=True):
    if npy: # file in numpy format
        data = np.load(filename)
    else:   # file in hdfs format
        data = hdf5_to_numpy(filename)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()


# Read in a numpy file with shape of (N, 3, 480, 480)
# Create the movie of a variable
def show_heatmap_animation(filename, save_path=None):
    fig = plt.figure()
    # Set margins, remove extra space
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
    ims = []

    data = np.load(filename)
    for i in range(data.shape[0]):
        im = plt.imshow(data[i, 0], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)

    if save_path is not None:
        '''
        matplotlib.use("Agg")
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
        '''
        ani.save(save_path, fps=20)

    #plt.show()



if __name__ == "__main__":
    if len(sys.argv) != 2 :
        sys.exit("Usage: python plot dataset_directory")
    path = sys.argv[1]
    #show_heatmap(path)
    show_heatmap_animation(path, "test.mp4")
