#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "omp.h"

#define MAX_VALUE_GEN 100
#define MIN_VALUE_GEN 1
#define ERROR_NUM_INPUT "Error : input previsti: \n" \
                        "1. Numero righe\n"         \
                        "2. Numero di colonne\n"\
                        "3. Numbero di threads\n"
#define ERROR_NUM_ROWS "Error : il numero di righe non puo' esser inferiore o uguale a zero\n"
#define ERROR_NUM_COLUMNS "Error : il numero di colonne non puo' esser inferiore o uguale a zero\n"
#define ERROR_NUM_THREAD "Errore : non e' possibile assegnare almeno una riga per thread. Riprovare incrementando il numero di righe\n"
#define ERROR_OPEN_FILE "Error : impossibile scrivere su file\n"
#define ERROR_LESS_THREAD "Error: indicare almeno due thread per svolgere l'operazione\n"

int check_input(int argc, char const *argv[], int *rows, int *columns,int *threads);
double *gen_random_array(int length,int threads);
double **gen_random_matrix(int rows, int columns,int threads);
double *matxvet(int rows, int columns, double *array, double **matrix,int threads);
double get_time(struct timeval time);
void check_input_condition(int boolean,char *error_code);
void print_array(double *array, int length,char* name);
void print_matrix(double **matrix, int rows, int columns);

int main(int argc, char const *argv[])
{
    int i, j, rows, columns, threads;
    struct timeval time;
    double inizio, fine;
    double **matrix;
    double *array;
    double *result;

    check_input(argc, argv, &rows, &columns,&threads);

    matrix = gen_random_matrix(rows, columns,threads);
    array = gen_random_array(columns,threads);

    inizio = get_time(time);
    result = matxvet(rows, columns, array, matrix,threads);
    fine = get_time(time);

    print_matrix(matrix,rows,columns);
    print_array(array,columns,"array");
    print_array(result,columns,"risultato");
    printf("tempo %f, con %d %d %d\n", (fine - inizio),rows,columns,threads);

    return 0;
}

/**
 * Verifica il numero di input, il valore da assegnare a rows, columns e numero thread. 
 * Inoltre verifica se è possibile assegnare almeno due righe per thread
 * @param argc numero di input main
 * @param argv array di input 
 * @param rows variabile in cui salvare il numero di righe
 * @param columns variabile in cui salvare il numero di colonne
 * @param threads variabile in cui salvare il numero di thread
*/
int check_input(int argc, char const *argv[], int *rows, int *columns,int *threads)
{
    check_input_condition(argc != 4, ERROR_NUM_INPUT);

    *rows = atoi(argv[1]);
    check_input_condition((*rows <= 0),ERROR_NUM_ROWS);

    *columns = atoi(argv[2]);
    check_input_condition((*columns <= 0),ERROR_NUM_COLUMNS);

    *threads = atoi(argv[3]);
    check_input_condition((*threads > (*rows / 2)),ERROR_NUM_THREAD);
    check_input_condition(*threads <= 1, ERROR_LESS_THREAD);

    omp_set_num_threads(*threads);
}

/**
 * @param boolean condizione che se verificata scatena una eccezione
 * @param error_code errore stampato in caso di eccezione
*/
void check_input_condition(int boolean, char *error_code){
    if( boolean == 1){
        printf("%s", error_code);
        exit(1);
    }
}

/**
 * @param length elementi da generare
 * @param thread numero di thread
 * @return vettore di length double generati casualmente
*/
double *gen_random_array(int length,int threads)
{
    int i;
    double *array;
    array = (double *)malloc(length* sizeof(double));
    #pragma omp parallel for default(none) shared(length, array) private(i) num_threads(threads)
    for (i = 0; i < length; i++)
    {
        array[i] = MIN_VALUE_GEN + (double)rand() / RAND_MAX * (MAX_VALUE_GEN - MIN_VALUE_GEN);
    }
    return array;
}

/**
 * @param rows righe della matrice
 * @param columns colonne della matrice
 * @param thread numero di thread
 * @return matrice di (rows x columns) double generati casualmente
*/
double **gen_random_matrix(int rows, int columns,int threads)
{
    int i, j;
    double **matrix;
    matrix = (double **)malloc(rows * sizeof(double *));
    #pragma omp parallel for default(none) shared(rows, columns, matrix) private(i, j) num_threads(threads)
    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double *)malloc(columns* sizeof(double));
        for (j = 0; j < columns; j++)
        {
            matrix[i][j] = MIN_VALUE_GEN + (double)rand() / RAND_MAX * (MAX_VALUE_GEN - MIN_VALUE_GEN);
        }
    }
    return matrix;
}

/**
 * Dati una matrice matrix ed un vettore array , applica il prodotto matrice-vettore
 * @param rows numero di righe della matrice
 * @param columns numero di colonne della matrice e di elementi del vettore
 * @param array vettore del prodotto
 * @param matrix matrice del prodotto
 * @param thread numero di thread
 * @return prodotto matrice-vettore
*/
double *matxvet(int rows, int columns, double *array, double **matrix,int threads)
{
    int i = 0, j = 0;
    double *result = (double *)malloc(rows* sizeof(double));
    #pragma omp parallel for default(none) shared(rows, columns, array, matrix, result) private(i, j) num_threads(threads)
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            result[i] += matrix[i][j] * array[j];
        }
    }
    return result;
}

/**
 * Restituisce il numero di secondi, con precisione al microsecondo , del tempo trascorso dall'esecuzione
 * @param time struct timeval su cui applicare gettimeofday e leggere il risultato
 * @return secondi + microsecondi trascorsi
*/
double get_time(struct timeval time)
{
    double result;
    gettimeofday(&time, NULL);
    result = time.tv_sec + (time.tv_usec / 1000000.0);
    return result;
}

/**
 * Scrive la matrice a video
 * @param matrix matrice da cui leggere i valori
 * @param rows numero righe
 * @param columns nome dell'array
*/
void print_matrix(double **matrix, int rows, int columns)
{
    int i, j;
    printf("\nMATRIX\n");
    for (i = 0; i < rows; i++)
    {
        printf("| ");
        for (j = 0; j < columns; j++)
        {
            printf( "%5.3f | ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Scrive l'array a video
 * @param array matrice da cui leggere i valori
 * @param length numero righe
 * @param name nome dell'array
*/
void print_array(double *array, int length,char* name)
{
    int i;
    printf( "\n%s\n|", name);
    for (i = 0; i < length; i++)
    {
        printf( " %5.3f |", array[i]);
    }
    printf("\n");
}