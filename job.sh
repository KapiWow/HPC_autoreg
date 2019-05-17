#!/bin/sh
#// -o ./res.out --nodes=1 --ntasks=9 --cpus-per-task=1
#SRUN -o ./res.out --nodes=1 --ntasks=8
export OMP_NUM_THREADS=8
time ./autoreg
