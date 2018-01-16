#!/bin/sh

#PBS -N HEAT_CLASSIFIER
#PBS -l nodes=1:ppn=16
#PBS -l walltime=01:59:00
#PBS -j oe

cd /home/wangchen/detector/heat_distribution_multi_iterations
python classifier.py
