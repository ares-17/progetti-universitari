#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "mpi.h"

void leggi_righe(int menum, int argc, char **argv, int *row);
int *getPeriod();
int *getNumeroElementiPerDimensione(int row, int col);

int const numero_dimensioni = 2;

int main(int argc, char **argv)
{
    int menum, nproc, row, col, menum_grid;
    int *num_elementi_per_dimensione, reorder, *period, *coordinate;

    MPI_Comm comm_grid;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &menum);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    leggi_righe(menum, argc, argv, &row);
    MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);

    col = (row / nproc) / numero_dimensioni;
    coordinate = (int *)calloc(numero_dimensioni, sizeof(int));
    num_elementi_per_dimensione = getNumeroElementiPerDimensione((row / nproc) / numero_dimensioni, col);
    printf("numero elementi pe dimensione %d %d\tcol:%d\n",num_elementi_per_dimensione[0], num_elementi_per_dimensione[1],col);
    period = getPeriod();
    reorder = 0;

    // Definisce una struttura logica dei processori , creando un nuovo communicator con regole di comunicazione ristrette
    MPI_Cart_create(MPI_COMM_WORLD, numero_dimensioni, num_elementi_per_dimensione, period, reorder, &comm_grid);
    MPI_Comm_rank(comm_grid, &menum_grid);
    // ritorna le coordinate della cella del processore con menum_grid
    MPI_Cart_coords(comm_grid, menum_grid, numero_dimensioni, coordinate);

    printf("Processore %d coordinate nella griglia(%d,%d) \n", menum, *coordinate, *(coordinate + 1));

    MPI_Finalize();
    return 0;
}

void leggi_righe(int menum, int argc, char **argv, int *row)
{
    if (argc != 2)
    {
        if (menum == 0)
        {
            printf("inserire il numero di righe\n");
        }
        MPI_Finalize();
    }
    else
    {
        *row = atoi(argv[1]);
    }
}

int *getPeriod()
{
    int *tmp = (int *)calloc(numero_dimensioni, sizeof(int));
    tmp[0] = tmp[1] = 0;
    return tmp;
}

int *getNumeroElementiPerDimensione(int row, int col)
{
    int *num_elementi_per_dimensione = (int *)calloc(numero_dimensioni, sizeof(int));
    num_elementi_per_dimensione[0] = row;
    num_elementi_per_dimensione[1] = col;
    return num_elementi_per_dimensione;
}
