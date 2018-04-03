#!/bin/sh

for error_iter in {0..199}; do
    T=$( expr $error_iter + 5 )
    echo $error_iter $T
    python heat_distribution.py -e $error_iter -t $T
done
