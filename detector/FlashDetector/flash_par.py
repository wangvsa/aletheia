'''
For generating flash.par files
1. randomly set initial conditions in flash.par
2. run the app to get data
'''

import sys
import os
import time
import random

def read_par_file(filename):
    pars = {}
    with open(filename) as lines:
        for line in lines:
            if "=" in line and not line.startswith("#"):
                line = "".join(line.split())    # remove all whitespaces, e.g. \t\n
                name = line.split("=")[0]
                value = line.split("=")[1]
                pars[name] = value
    return pars

def write_par_file(filename, pars):
    with open(filename, "w") as f:
        for key, value in pars.iteritems():
            line = key + "=" + str(value)
            f.write(line+"\n")

def get_random_pars(app):
    # Sod
    if app == "Sod":
        modify_pars = [("sim_rhoLeft", 0.0, 5.0), ("sim_rhoRight", 0.0, 5.0),
         ("sim_pLeft", 0, 5.0), ("sim_pRight", 0.0, 5.0),
         ("sim_uLeft", 0, 5.0), ("sim_uRight", 0.0, 5.0)]

    # Blast2
    if app == "Blast2":
        modify_pars = [("sim_rhoLeft", 0.0, 5.0), ("sim_rhoMid", 0.0, 5.0),("sim_rhoRight", 0.0, 5.0),
         ("sim_pLeft", 0, 1000.0), ("sim_pMid", 0.0, 200), ("sim_pRight", 0.0, 200),
         ("sim_uLeft", 0, 5.0), ("sim_uMid", 0.0, 5.0), ("sim_uRight", 0.0, 5.0)]

    # BlastBS
    if app == "BlastBS":
        modify_pars = [("gamma", 1.0, 3.0), ("xCtr", 0.0, 5.0),
         ("yCtr", 0, 5.0), ("Radius", 0.0, 1.0), ("Bx0", 50, 200)]

    # BrioWu
    if app == "BrioWu":
        modify_pars = [("rho_left", 0.0, 2), ("rho_right", 0.0, 2),
         ("p_left", 0, 2), ("p_right", 0.0, 2), ("u_left", 0, 2), ("u_right", 0.0, 2),
         ("v_left", 0, 2), ("v_right", 0.0, 2), ("w_left", 0, 2), ("w_right", 0.0, 2)]

    # Sedov
    if app == "Sedov":
        modify_pars = [("sim_pAmbient", 0.0, 0.001), ("sim_rhoAmbient", 0.5, 3), ("sim_expEnergy", 0.5, 3),
         ("sim_rInit", 0, 0.2), ("sim_xctr", 0.0, 2), ("sim_yctr", 0, 2), ("sim_zctr", 0.0, 2)]

    return modify_pars


'''
Generate random par files
Creeate a directory for each initial condition
'''
def generate_random_pars(idx):
    app_name = sys.argv[1]
    app_dir = sys.argv[2] + "/"
    org_par_file = app_dir + "flash.par"

    pars = read_par_file(org_par_file)
    modified_pars = get_random_pars(app_name)

    for par in modified_pars:
        pars[par[0]] = random.uniform(par[1], par[2])
    #pars['output_directory'] = "\"./data/"+"\""

    random_data_dir  = app_dir + "data_" + str(idx)
    new_par_path = random_data_dir + "/flash.par"
    new_flash_path = random_data_dir + "/flash4"

    # Create directory and write the randomly generated new par file
    os.system("mkdir "+ random_data_dir)
    write_par_file(new_par_path, pars)

    # Run Flash with the random initial conditions
    os.system("cp "+app_dir+"flash4 "+new_flash_path)
    os.system("cd "+random_data_dir+"&& mpirun -np 8 "+new_flash_path+" -par_file "+new_par_path)
    #os.system("cd "+random_data_dir+"&& aprun -n 8 -N 8 "+new_flash_path+" -par_file "+new_par_path)


'''
Usage:
python flash_par.py app_name app_dir

This will create app_dir/data_0, app_dir/data_1, ...,
where each directory contains a randomly generated par file

Then the program will run under each of these directories to
generate clean checkpoint data
'''
if __name__ == "__main__" :
    for i in range(2):
        generate_random_pars(i)
