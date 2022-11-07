#!/bin/bash

export OMP_NUM_THREADS=5
gcc  -fopenmp -o somma\ numeri somma\ numeri.c
./somma\ numeri