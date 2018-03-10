'''
Generate flash.par to restart from a corrupted checkpoint
'''
import h5py
import random
import glob
import os
from bits import bit_flip
import time
import math
import sys
import numpy as np
import flash_par

FLASH_APPS = ["Blast2", "BlastBS", "BrioWu", "Cellular",
                "DMReflection", "RHD_Sod", "Sedov", "Sod"]

def get_flip_error(val):
    while True:
        pos = random.randint(41, 50)
        error =  bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    #error = min(10e+3, error)
    #error = max(-10e+3, error)
    #if error > 0 and error < 1e-5: error = 0
    return error

'''
Copy par file
Modify it so we can restart from the given corrupted checkpoint
Also set the nend to 20 timesteps after restarting
'''
def modify_par_file(data_dir, basenm, restart_point):
    par_file = data_dir + "flash.par"

    pars = flash_par.read_par_file(par_file)
    pars['basenm'] = "\""+basenm+"\""
    pars['nend'] = str(restart_point+19)
    pars['checkpointFileNumber'] = str(restart_point)
    pars['restart'] = ".true."

    # output plot file is enough
    # no need to ouput checkpoint file
    pars['checkpointFileIntervalStep'] = "0"
    pars['plotfileIntervalStep'] = "5"

    new_par_file = par_file + "." + basenm[:-1] + ".new_par" # remove the last _, "new_par" just easy for my pbs script to handle
    flash_par.write_par_file(new_par_file, pars)
    return new_par_file


'''
Copy a clean checkpoint
Insert an error into the copied file
Return the basenm which contains the error info: restart_point,x,y,bit
'''
def insert_errors(data_dir, restart_point):

    filenumber = ("0000"+str(restart_point))[-4:]
    basenm = "error_%s_" %(restart_point) + str(int(time.time()*1000))
    clean_checkpoint_file = data_dir + "sod_hdf5_chk_"+filenumber
    corrupted_checkpoint_file = data_dir + basenm + "hdf5_chk_" + filenumber

    # Copy a corrupted checkpoint file
    os.system("cp "+clean_checkpoint_file+" "+corrupted_checkpoint_file)
    f = h5py.File(corrupted_checkpoint_file, 'r+')
    f_org = f['dens'][:].copy()

    # Inset one into each of the windows of a 480x480 frame,
    # so we will have 11x11 windows (60, 60, overlap:20)
    valid, all = 0, 0
    for start_y in range(20, 460, 40):
        for start_x in range(20, 460, 40):
            x = start_x + random.randint(0, 20)
            y = start_y + random.randint(0, 20)
            # Insert error
            org = f['dens'][0, 0, y, x]
            error = get_flip_error(org)

            threshold = (np.max(f_org) - np.min(f_org)) * 0.00078125
            if abs(error - org) > threshold: valid = valid + 1
            all = all+1

            f['dens'][0, 0, y, x] = error

    f.close()
    print "valid:", valid, ", all:", all
    return basenm, valid, all

def restart(data_dir):
    valid, all = 0, 0
    for restart_point in range(0, 200, 20): # 0~200, step=20
        basename, t1, t2 = insert_errors(data_dir, restart_point)
        valid = valid + t1
        all = all + t2
        new_par_file = modify_par_file(data_dir, basename, restart_point)
        print(new_par_file)
    print valid, all


# Usage python flash_restart.sh data_dir
if __name__ == "__main__":
    # create restart par file from different initial condtions
    for data_dir in glob.glob(sys.argv[1]):
        restart(data_dir)
