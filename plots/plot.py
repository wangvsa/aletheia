import sys
import h5py
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read from a hdf5 file to a numpy array
def hdf5_to_numpy(filename):
    f = h5py.File(filename, 'r')
    data = f[list(f.keys())[0]][:]
    return data

def show_heatmap(filename):
    data = hdf5_to_numpy(filename)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()


def show_heatmap_animation(dataset_dir, save_path=None):
    fig = plt.figure()
    ims = []
    for filename in glob.iglob(dataset_dir+"/*.h5"):
        data = hdf5_to_numpy(filename)
        im = plt.imshow(data, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)

    if save_path is not None:
        matplotlib.use("Agg")
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)

    plt.show()



if __name__ == "__main__":
    path = sys.argv[1]
    #show_heatmap(path)
    show_heatmap_animation(path)
