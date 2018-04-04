#!/bin/sh

for error_iter in {0..199}; do
    T=$( expr $error_iter + 100)
    echo $error_iter $T
    python heat_distribution.py -e $error_iter -t $T -i 5
done
