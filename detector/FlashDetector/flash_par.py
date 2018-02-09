import commands
import h5py
import random
import glob
import os
from bits import bit_flip
import math

SOD_DIR = "/home/wangchen/Sources/FLASH4.4/objects/Sod_multi/"
FLASH_APPS = ["Blast2", "BlastBS", "BrioWu", "Cellular",
                "DMReflection", "RHD_Sod", "Sedov", "Sod"]

par_file = SOD_DIR + "flash.par"

# delete last two lines
def delete_last_lines():
    commands.getstatusoutput("head -n -2 " + par_file + " > tmp.par")
    commands.getstatusoutput("mv tmp.par " + par_file)

def append_to_par(restart_point):
    # Specify restart point
    commands.getstatusoutput("echo checkpointFileNumber = " + str(restart_point) + " >> " + par_file)
    # Specify end point
    commands.getstatusoutput("echo nend = " + str(restart_point+10) + " >> " + par_file)


def insert_error(restart_point):
    filenumber = ("0000"+str(restart_point))[-4:]
    checkpoint_file = SOD_DIR + "clean_checkpoints/sod_hdf5_chk_"+filenumber
    commands.getstatusoutput("cp "+checkpoint_file+" "+SOD_DIR)
    checkpoint_file = SOD_DIR + "sod_hdf5_chk_"+ filenumber
    f = h5py.File(checkpoint_file, 'r+')

    blockId = random.randint(0, f['dens'].shape[0]-1)
    x = random.randint(0, 7)
    y = random.randint(0, 7)
    org = f['dens'][blockId, 0, y, x]

    bit = random.randint(1, 10)
    error = bit_flip(org, bit)

    if math.isnan(error) or math.isinf(error):
        error = org * 5
    if error > org * 5:
        error = org * 5
    f['dens'][blockId, 0, y, x] = error
    f['dens'][blockId, 0, y, x] = 5*org
    coordinate = list(f['coordinates'][blockId])
    f.close()

    basename = "sod_"+str([restart_point, blockId, x, y])+"_"+str(coordinate)+"_"
    return basename


def rename_plot_files(basename):
    for filename in glob.iglob(SOD_DIR+"sod_hdf5_plt_cnt*"):
        newname = SOD_DIR + basename + filename.split("/")[-1]
        os.rename(filename, newname)


def test():
    # Restart from checkpoint i
    for restart_point in range(1000):
        if restart_point == 100:
        #if (restart_point-15) % 50 == 0:
            # Insert error
            basename = insert_error(restart_point)

            # Modify par file to let it restart from given checkpoint
            delete_last_lines()
            append_to_par(restart_point)

            # Restart the program
            cmd = "cd "+ SOD_DIR +" && mpirun -np 8 ./flash4"
            status = commands.getstatusoutput(cmd)[0]
            rename_plot_files(basename)

            # CD back
            commands.getstatusoutput("cd /home/wangchen/Sources/aletheia/detector/FlashDetector/")

            print "restart at:", restart_point, ", status:", status
            print basename

for i in range(1000):
    test()
