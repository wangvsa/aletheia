#!/bin/sh

#PBS -N CFD_DETECTOR
#PBS -l nodes=1:ppn=16
#PBS -l walltime=03:59:00
#PBS -j oe

cd /home/wangchen/cfd
python cfd_detector.py
