import h5py
import glob
import numpy as np

for i in range(99):
    prefix = "error_90_72_264_263hdf5_plt_cnt_"
    error_filename = prefix + ("0000"+str(i))[-4:]

    clean_filename = "./clean/blast2_hdf5_chk_" + ("0000"+str(91+i))[-4:]
    print(clean_filename, error_filename)

    clean_file = h5py.File(clean_filename)
    error_file = h5py.File(error_filename, 'r+')    # r+ has permission to write

    error_file['dens'][:] = np.abs(error_file['dens'][:] - clean_file['dens'][:])

    clean_file.close()
    error_file.close()
