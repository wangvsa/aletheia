import h5py



'''
Read hdf5 data file
args:
    prefix: prefix for each data file
    it: which iteration's data to read
'''
def read_data(prefix, it, var_name="dens"):
    filename = prefix + ("0000"+str(it))[-4:]
    f = h5py.File(filename)
    data = f[var_name][0,0]     # shape of (nx, ny)
    return data


