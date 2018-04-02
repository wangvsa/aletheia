import sys
import glob
import numpy as np

# Combine all npy files in a given directory
def combine_npys(path):
    dataset = []
    for filename in glob.iglob(path+"/*.npy"):
        data = np.load(filename)
        print(filename, data.shape)
        dataset.append(np.load(filename))
    dataset = np.vstack(dataset)
    print("save combined data: ", dataset.shape)
    np.save("combined.npy", dataset)

combine_npys(sys.argv[1])
