import sys
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

def get_random_pars():
    modify_pars = [("sim_rhoLeft", 0.0, 5.0), ("sim_rhoRight", 0.0, 5.0),
     ("sim_pLeft", 0, 5.0), ("sim_pRight", 0.0, 5.0),
     ("sim_uLeft", 0, 5.0), ("sim_uRight", 0.0, 5.0)]
    return modify_pars

def main(filename):
    pars = read_par_file(filename)
    modify_pars = get_random_pars()

    for par in modify_pars :
        pars[par[0]] = random.uniform(par[1], par[2])

    write_par_file(filename, pars)



if __name__ == "__main__" :
    main(sys.argv[1])
