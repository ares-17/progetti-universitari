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
#define ERROR_NUM_ROWS "Error : il numero di righe non puo' esser inferiore a zero\n"
#define ERROR_NUM_COLUMNS "Error : il numero di colonne non puo' esser inferiore a zero\n"
#define ERROR_NUM_THREAD "Errore : non e' possibile assegnare almeno una riga per thread. Riprovare incrementando il numero di righe"
#define ERROR_OPEN_FILE "Error : impossibile scrivere su file\n"

int check_input(int argc, char const *argv[], int *rows, int *columns,int *threads);
FILE *create_file();
double *gen_random_array(int length,int threads);
double **gen_random_matrix(int rows, int columns,int threads);
double *matxvet(int rows, int columns, double *array, double **matrix,int threads);
double get_time(struct timeval time);
void print_matrix(double **matrix, int rows, int columns, FILE *file);
void print_array(double *array, int length, FILE *file, char *name);


int main(int argc, char const *argv[])
{
    int i, j, rows, columns, threads;
    struct timeval time;
    double inizio, fine;
    double **matrix;
    double *array;
    double *result;
    FILE *file;

    check_input(argc, argv, &rows, &columns,&threads);
    file = create_file();

    matrix = gen_random_matrix(rows, columns,threads);
    print_matrix(matrix, rows, columns, file);
    array = gen_random_array(rows,threads);
    print_array(array, rows, file, "array");

    inizio = get_time(time);
    result = matxvet(rows, columns, array, matrix,threads);
    fine = get_time(time);

    print_array(result, rows, file, "result");
    printf("tempo %f, con %d %d %d\n", (fine - inizio),rows,columns,threads);

    fclose(file);

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
    if (argc != 4)
    {
        printf("%s", (ERROR_NUM_INPUT));
        exit(-1);
    }

    *rows = atoi(argv[1]);
    if (*rows <= 0)
    {
        printf("%s", (ERROR_NUM_ROWS));
        exit(-1);
    }
    *columns = atoi(argv[2]);
    if (*columns <= 0)
    {
        printf("%s", (ERROR_NUM_COLUMNS));
        exit(-1);
    }
    *threads = atoi(argv[3]);
    if (*threads > (*rows / 2))
    {
        printf("%s", (ERROR_NUM_THREAD));
        exit(-1);
    }
    omp_set_num_threads(*threads);
}

/**
 * Crea un file secondo il nome 'elaborato_2', con opzione 'w'
 * @return file creato
*/
FILE *create_file()
{
    FILE *file;
    if ((file = fopen("elaborato_2.log", "a+")) == NULL)
    {
        printf(ERROR_OPEN_FILE);
        exit(1);
    }
    return file;
}

/**
 * @param length elementi da generare
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
        //array[i] = MIN_VALUE_GEN + (double)rand() / RAND_MAX * (MAX_VALUE_GEN - MIN_VALUE_GEN);
        array[i] = 1;
    }
    return array;
}

/**
 * @param rows righe della matrice
 * @param columns colonne della matrice
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
            //matrix[i][j] = MIN_VALUE_GEN + (double)rand() / RAND_MAX * (MAX_VALUE_GEN - MIN_VALUE_GEN);
            matrix[i][j] = j + i;
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
 * Scrive la matrice ,formattata , sul file passato in input
 * @param matrix matrice da cui leggere i valori
 * @param rows numero righe
 * @param columns numero colonne
 * @param file file su cui scrivere, si presuppone già aperto
*/
void print_matrix(double **matrix, int rows, int columns, FILE *file)
{
    int i, j;
    fputs("\nMATRIX\n", file);
    for (i = 0; i < rows; i++)
    {
        fputs("| ", file);
        for (j = 0; j < columns; j++)
        {
            fprintf(file, "%5.2f | ", matrix[i][j]);
        }
        fputs("\n", file);
    }
    fputs("\n", file);
}

/**
 * Scrive l'array  ,formattato , sul file passato in input
 * @param array matrice da cui leggere i valori
 * @param length numero righe
 * @param file file su cui scrivere, si presuppone già aperto
 * @param name nome dell'array
*/
void print_array(double *array, int length, FILE *file, char *name)
{
    int i;
    fprintf(file, "\n%s\n|", name);
    for (i = 0; i < length; i++)
    {
        fprintf(file, " %5.2f |", array[i]);
    }
    fputs("\n", file);
}
