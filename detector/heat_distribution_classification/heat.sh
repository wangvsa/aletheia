#!/bin/sh

#PBS -N HEAT_CNN2
#PBS -l nodes=1:ppn=16
#PBS -l walltime=03:59:00
#PBS -j oe

cd /home/wangchen/detector/heat_distribution_classification
python classifier.py
