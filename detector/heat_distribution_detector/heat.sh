#!/bin/sh

#PBS -N HEAT_DETECTOR
#PBS -l nodes=1:ppn=16
#PBS -l walltime=02:59:00
#PBS -j oe

cd /home/wangchen/detector/heat_distribution_detector
python detector.py
