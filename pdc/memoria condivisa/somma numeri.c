#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "omp.h"
#include <time.h>

double get_time(struct timeval time);
void semplice_parallelizzazione();
void ciclo_parallelo();
void ciclo_parallelo_firstprivate();
void ciclo_parallelo_lastprivate();
void ciclo_parallelo_reduction();

int main(int argc, char const *argv[])
{
    ciclo_parallelo_reduction();
    return 0;
}

double get_time(struct timeval time)
{
    double result;
    gettimeofday(&time, NULL);
    result = time.tv_sec + (time.tv_usec / 1000000.0);
    return result;
}

void semplice_parallelizzazione()
{
#pragma omp parallel // oppure #pragma omp for , con un ciclo che segue. L'esecuzione rimane sempre sequenziale
    {
        printf("Hello from process %d\n", omp_get_thread_num());
    }
}

/**
 * Se sono presenti variabili modificate da uno o più thread nel ciclo e ne viene letto il valore dopo
 * la parallelizzazione, i valori delle variabili avranno:
 *  1. il valore dell'ultimo thread che ha modificato
 *  2. il valore dell'ultimo thread in vita
 * Le variabili definite all'interno dei cicli sono automaticamente private
*/
void ciclo_parallelo()
{
#pragma omp parallel for
    for (int i = 0; i < 10; i++)
    {
        printf("Hello from process %d\n", omp_get_thread_num());
    }
}

/**
 * Con 'firstprivate()' per le variabili presenti nella lista sono assegnate a TUTTI i processi delle copie, poi deallocate a fine esecuzione.
 * Difatti se mancasse tale clausola, l'ultimo thread che modifica le variabili indicate, sovrascriverebbe il valore globale di queste
*/
void ciclo_parallelo_firstprivate()
{
    int random = 10;
    printf("Prima del ciclo indirizzo %d, valore %d\n",&random, random);
    #pragma omp parallel for firstprivate(random)
    for (int i = 0; i < omp_get_num_threads() ; i++)
    {
        srand(time(NULL));
        random += i;
        if(omp_get_thread_num() == 1){
            printf("dal secondo thread, il valore %d\n",random);
        }
    }

    printf("Dopo del ciclo indirizzo %d, valore %d\n",&random, random);
}

/**
 * la clausola 'lastprivate' assicura che il thread con indice del ciclo maggiore (l'ultimo) assegni il valore alla variabile.
 * Alla variabile è modificato solo il valore, non è re-inizializzata con un altro indirizzo
 * TODO: perche usare sia lastprivate che firstprivate
*/
void ciclo_parallelo_lastprivate(){

    int random = 0;
    printf("Prima del ciclo indirizzo %d, valore %d\n",&random, random);
    #pragma omp parallel for firstprivate(random) lastprivate(random)
    for (int i = 0; i < omp_get_num_threads() ; i++)
    {
        srand(time(NULL));
        random = i;
        printf("Thread %d: indirizzo %d, valore %d\n",omp_get_thread_num(),&random, random);
        if(omp_get_thread_num() == 2){
            printf("hello ! motherfuckker!!\n");
        }
    }
    printf("Dopo del ciclo indirizzo %d, valore %d\n",&random, random);
}

/**
 * la clausola 'reduction' assicura che sia allocata una copia per ogni thread e l'operazione compiuta
 * (che deve essere specificata nella clausola) è thread-safe
*/
void ciclo_parallelo_reduction(){
    int random = 0;
    printf("Prima del ciclo indirizzo %d, valore %d\n",&random, random);
    #pragma omp parallel for reduction(+:random)
    for (int i = 0; i < omp_get_num_threads() ; i++)
    {
        srand(time(NULL));
        random += i;
        printf("Thread %d: indirizzo %d, valore %d\n",omp_get_thread_num(),&random, random);
        if(omp_get_thread_num() == 2){
            printf("hello ! motherfuckker!!\n");
        }
    }
    printf("Dopo del ciclo indirizzo %d, valore %d\n",&random, random);
}