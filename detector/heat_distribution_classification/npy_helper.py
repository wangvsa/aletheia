import sys
import glob
import numpy as np

# Combine all npy files in a given directory that meet regex
def combine_npys(regex, output_file):
    dataset = []
    for filename in glob.iglob(regex):
        data = np.load(filename)
        print(filename, data.shape)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0) # make (480, 480) -> (1, 480, 480)
        dataset.append(data)
    dataset = np.vstack(dataset)
    print("save to %s, shape %s" %(output_file, dataset.shape))
    np.save(output_file, dataset)


# Combine all npys files of one iteration
def combine_one_iteration(data_dir):
    for i in range(0, 100, 5):
        regex = data_dir + "/*_"  + str(i) + ".npy" # e.g. *_10.npy
        print(regex)
        combine_npys(regex, "after_"+str(i))


# Combine all npys
#combine_npys(sys.argv[1]+"/*.npy", "combined")
combine_one_iteration(sys.argv[1])
