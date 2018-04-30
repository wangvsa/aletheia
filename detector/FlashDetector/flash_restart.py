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
                "blastBS_mhdeflection", "RHD_Sod", "Sedov", "Sod"]

def get_flip_error(val):
    while True:
        pos = random.randint(1, 20)
        error =  bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    error = min(10e+3, error)
    error = max(-10e+3, error)
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
    pars['nend'] = str(restart_point+99)
    pars['checkpointFileNumber'] = str(restart_point)
    pars['restart'] = ".true."

    # output plot file is enough
    # no need to ouput checkpoint file
    pars['checkpointFileIntervalStep'] = "0"
    pars['plotfileIntervalStep'] = "1"

    new_par_file = data_dir + "/flash." + basenm + ".new_par" # remove the last _, "new_par" just easy for my pbs script to handle
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
    clean_data_dir =  data_dir + "/clean/"
    clean_checkpoint_file = clean_data_dir + "conduction_hdf5_chk_"+filenumber
    corrupted_checkpoint_file = data_dir + basenm + "hdf5_chk_" + filenumber

    # Copy a corrupted checkpoint file
    os.system("cp "+clean_checkpoint_file+" "+corrupted_checkpoint_file)
    f = h5py.File(corrupted_checkpoint_file, 'r+')
    f_org = f['dens'][:].copy()

    # Insert one into each of the windows of a 480x480 frame,
    # so we will have 11x11 windows (60, 60, overlap:20)
    for start_y in range(20, 460, 40):
        for start_x in range(20, 460, 40):
            x = start_x + random.randint(0, 20)
            y = start_y + random.randint(0, 20)
            # Insert error
            old = f['dens'][0,0,y,x]
            f['dens'][0, 0, y, x] = get_flip_error(f['dens'][0,0,y,x])
            #print("(%s,%s, old:%s, new:%s)" %(x,y,old,f['dens'][0,0,y,x]))
    f.close()
    return basenm

# Only insert one error into one checkpoint
def insert_error(data_dir, restart_point):

    win_x, win_y = random.randint(0, 11-1), random.randint(0, 11-1)
    x = np.arange(20, 460, 40)[win_x] + random.randint(0, 20)
    y = np.arange(20, 460, 40)[win_y] + random.randint(0, 20)
    error_win_id = win_x * 11 + win_y

    filenumber = ("0000"+str(restart_point))[-4:]
    basenm = "error_%s_%s_%s_%s" %(restart_point, error_win_id, x, y)

    clean_data_dir =  data_dir + "/clean/"
    clean_checkpoint_file = clean_data_dir + "conduction_hdf5_chk_"+filenumber

    corrupted_checkpoint_file = data_dir + basenm + "hdf5_chk_" + filenumber

    # Copy a corrupted checkpoint file
    os.system("cp "+clean_checkpoint_file+" "+corrupted_checkpoint_file)
    f = h5py.File(corrupted_checkpoint_file, 'r+')
    print("old:", f['dens'][0,0,x,y])
    f['dens'][0, 0,x,y] = get_flip_error(f['dens'][0,0,x,y])
    print("new:", f['dens'][0,0,x,y])

    f.close()
    return basenm

def restart(data_dir):
    for restart_point in range(50, 100, 10): # 0~200, step=20
        basename = insert_error(data_dir, restart_point)
        #basename = insert_errors(data_dir, restart_point)
        new_par_file = modify_par_file(data_dir, basename, restart_point)
        print(new_par_file)


# Usage python flash_restart.sh data_dir
if __name__ == "__main__":
    # create restart par file from different initial condtions
    for data_dir in glob.glob(sys.argv[1]):
        restart(data_dir)
