import sys
import glob
import numpy as np

# Combine all npy files in a given directory
def combine_npys(path, regex=".npy"):
    dataset = []
    for filename in glob.iglob(path+"/*"+regex):
        data = np.load(filename)
        print(filename, data.shape)
        dataset.append(np.load(filename))
    dataset = np.vstack(dataset)
    print("save combined data: ", dataset.shape)
    np.save("after"+regex, dataset)

combine_npys(sys.argv[1], "_0.npy")
combine_npys(sys.argv[1], "_1.npy")
combine_npys(sys.argv[1], "_2.npy")
combine_npys(sys.argv[1], "_3.npy")
combine_npys(sys.argv[1], "_4.npy")
combine_npys(sys.argv[1], "_5.npy")
combine_npys(sys.argv[1], "_6.npy")
combine_npys(sys.argv[1], "_7.npy")
combine_npys(sys.argv[1], "_8.npy")
combine_npys(sys.argv[1], "_9.npy")
combine_npys(sys.argv[1], "_10.npy")
combine_npys(sys.argv[1], "_11.npy")
combine_npys(sys.argv[1], "_12.npy")
combine_npys(sys.argv[1], "_13.npy")
combine_npys(sys.argv[1], "_14.npy")
combine_npys(sys.argv[1], "_15.npy")
combine_npys(sys.argv[1], "_16.npy")
combine_npys(sys.argv[1], "_17.npy")
combine_npys(sys.argv[1], "_18.npy")
combine_npys(sys.argv[1], "_19.npy")
