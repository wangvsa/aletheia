#!/bin/sh

#PBS -N CFD_BIN_CLASSIFER
#PBS -l nodes=1:ppn=16
#PBS -l walltime=03:59:00
#PBS -j oe

cd /home/wangchen/detector/cfd_bin_classification
python cfd_classifier.py
