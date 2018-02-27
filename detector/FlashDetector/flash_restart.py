'''
Generate flash.par to restart from a corrupted checkpoint
'''
import h5py
import random
import glob
import os
from bits import bit_flip
import math
import flash_par

FLASH_APPS = ["Blast2", "BlastBS", "BrioWu", "Cellular",
                "DMReflection", "RHD_Sod", "Sedov", "Sod"]

BASE_DIR = "/home/wangchen/tmp/data_0/"
par_file = BASE_DIR + "flash.par"


'''
Copy par file
Modify it so we can restart from the given corrupted checkpoint
Also set the nend to 10 timesteps after restarting
'''
def modify_par_file(basenm, restart_point):
    pars = flash_par.read_par_file(par_file)
    pars['basenm'] = "\""+basenm+"\""
    pars['nend'] = str(restart_point+10)
    pars['checkpointFileNumber'] = str(restart_point)
    pars['restart'] = ".true."
    new_par_file = par_file + "." + basenm[:-1] # remove the last _
    flash_par.write_par_file(new_par_file, pars)
    return new_par_file


'''
Copy a clean checkpoint
Insert an error into the copied file
Return the basenm which contains the error info: restart_point,x,y,bit
'''
def insert_error(restart_point):
    x = random.randint(0, 1000-1)
    y = random.randint(0, 1000-1)
    bit = random.randint(0, 10-1)


    filenumber = ("0000"+str(restart_point))[-4:]
    basenm = "error_%s_%s_%s_%s_" %(restart_point, x, y, bit)
    clean_checkpoint_file = BASE_DIR + "hdf5_chk_"+filenumber
    corrupted_checkpoint_file = BASE_DIR+basenm+"hdf5_chk_"+filenumber

    # Copy a corrupted checkpoint file
    os.system("cp "+clean_checkpoint_file+" "+corrupted_checkpoint_file)
    f = h5py.File(corrupted_checkpoint_file, 'r+')

    # Insert error
    org = f['dens'][0, 0, y, x]
    error = bit_flip(org, bit)

    if math.isnan(error) or math.isinf(error):
        error = org * 1000
    if error > org * 1000: error = 1000*org
    f['dens'][0, 0, y, x] = error
    f.close()

    return basenm

def test():
    # Restart from checkpoint i
    for restart_point in range(10):
        if restart_point == 5:

            basename = insert_error(restart_point)

            new_par_file = modify_par_file(basename, restart_point)

            # Restart the program
            cmd = "cd "+ BASE_DIR +" && mpirun -np 8 ./flash4 -par_file "+new_par_file
            os.system(cmd)

test()
