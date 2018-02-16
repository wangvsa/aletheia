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

    if app == "BrioWu":
        modify_pars = [("rho_left", 0.0, 2), ("rho_right", 0.0, 2),
         ("p_left", 0, 2), ("p_right", 0.0, 2), ("u_left", 0, 2), ("u_right", 0.0, 2),
         ("v_left", 0, 2), ("v_right", 0.0, 2), ("w_left", 0, 2), ("w_right", 0.0, 2)]

    if app == "Sedov":
        modify_pars = [("sim_pAmbient", 0.0, 0.001), ("sim_rhoAmbient", 0.5, 3), ("sim_expEnergy", 0.5, 3),
         ("sim_rInit", 0, 0.2), ("sim_xctr", 0.0, 2), ("sim_yctr", 0, 2), ("sim_zctr", 0.0, 2)]

    return modify_pars


def run_flash_with_random_setting(app, flash_path, par_path):
    pars = read_par_file(par_path)
    modify_pars = get_random_pars(app)

    for par in modify_pars :
        pars[par[0]] = random.uniform(par[1], par[2])

    # change basenm and output directory
    pars['basenm'] = "\""+ str(int(time.time())) + "\""
    #pars['output_directory'] = "\"./data/train/"+app+"\""
    pars['output_directory'] = "\"./data/"+"\""

    write_par_file(par_path, pars)

    # Run Flash with the random initial conditions
    os.system("mpirun -np 8 "+flash_path+" -par_file "+par_path)


if __name__ == "__main__" :
    flash_path = sys.argv[1] + "/flash4"
    par_path = sys.argv[1] + "/flash.par"

    app = sys.argv[2]
    for _ in range(1):
        run_flash_with_random_setting(app, flash_path, par_path)
