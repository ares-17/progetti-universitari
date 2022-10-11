#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#define STRATEGY_1 1
#define STRATEGY_2 2
#define STRATEGY_3 3
#define TAG_STRATEGY(i) 77 + i
#define ARE_FEW_ELEMENTS(num_items, num_proc) (num_items / 2) < num_proc
#define IS_STRATEGY_OUT_OF_RANGE(strategy) strategy<1 || strategy> 3
#define ERROR_NUM_INPUT "Error : the program needs at least 5 inputs :\n" \ 
    "1st indicates the number of items to be added\n"                 \
    "2nd indicates the strategy\n"                                    \
    "3rd indicates the ID of the process that prints the total sum\n" \
    "and at least two elements to be added\n"
#define ERROR_RANDOM_VALUE "if you want to generate an N number of random elements, then enter :\n" \
                           "1^ as the number N of elements to sum up\n"                             \
                           "2^ the strategy\n"                                                      \
                           "3^ the id of the process that prints the result\n\n"
#define ERROR_STRATEGY "The strategy can be a number between 1 and 3\n"
#define WARNING_ID_STRATEGY "If first or second strategy is selected,\n" \
                            "only the process with an id equal to 0\n"   \
                            "has the total result of the operation!,\n"  \
                            "for this the id is placed at 0...\n\n"
#define ERROR_ITEMS_PROC "specify at least two elements to be summed per process !\n"
#define ERROR_NUM_PROC_STRATEGY "The algorithms of strategies 2 and 3\n"          \
                                "are only applicable on a number of processors\n" \
                                "that is power of 2 !\n\n"
#define ERROR_ID_RANGE "The specified processor id must be between 0 and the number of processors -1 !\n" \
                       " or equal to -1 to specify that all processors should print the final result\n\n"
#define WARNING_NPROC_ONE "Having specified only one processor,\n" \
                          "the MPI library will not be used and\n" \
                          "the entire calculation will be performed sequentially ...\n\n"
#define ERROR_NUM_INPUT_WITH_ARGS "The specified number of elements does not really\n" \
                                  "correspond to the number of elements passed as input!\n\n"
#define DISTRIBUTION_TAG(i) 501 + i
#define MIN_TO_GEN_RANDOM_VALUES 21
#define MAX_VALUE_GEN 100
#define MIN_VALUE_GEN 1

int const MIN_ARG_REQUIRED = 6;
int num_procs;

void read_input(char **argv, int *strategy, int *id);
int is_power_of_two(int x);
void warnings(int strategy, int *id);
void check_input(int memum, int *exit_status, int argc, int *strategy, char **argv, int *num_items_input, int *id);
void init_MPI(int num_elem, char **data, int *memum);
void calculate_elem_proc(int memum, int num_items_specified, int *num_data_proc, int *rest);
void read_performance(double start_time_proc, int memum);
void exponentials(int **exp2);
void parse_input(char **argv, int memum, double **data, int num_items_specified);
void distribute_data(int memum, int num_data_proc, int rest, double *data);
void get_data(int memum, int num_data_proc, double **data);
void start_performance(double *start_time);
void local_calculation(double *data, int num_data_proc, double *result);
void first_strategy(int memum, double *local_sum);
void second_strategy(int memum, double *partial_sum);
void third_strategy(int memum, double *partial_sum);
void communication_strategy(int strategy, int memum, double *local_sum);
void print_result(int memum, int id, int strategy, double sum);
void execute_seq(char **argv, int num_items);

int main(int argc, char **argv)
{
    int memum;
    int num_data_proc;
    int rest;
    double *data;
    double local_sum;
    double start_time;
    int exit_status = 0;
    int strategy = 0;
    int num_items_input = 0;
    int id = 0;

    init_MPI(argc, argv, &memum);
    check_input(memum, &exit_status, argc, &strategy, argv, &num_items_input, &id);

    if (exit_status != 0)
    {
        MPI_Finalize();
        return 0;
    }
    else if (num_procs == 1)
    {
        execute_seq(argv, num_items_input);
        MPI_Finalize();
        return 0;
    }

    parse_input(argv, memum, &data, num_items_input);
    calculate_elem_proc(memum, num_items_input, &num_data_proc, &rest);
    distribute_data(memum, num_data_proc, rest, data);
    get_data(memum, num_data_proc, &data);
    start_performance(&start_time);
    local_calculation(data, num_data_proc, &local_sum);
    communication_strategy(strategy, memum, &local_sum);
    print_result(memum, id, strategy, local_sum);
    read_performance(start_time, memum);

    MPI_Finalize();
}

/**
 * Deriva i valori da assegnare a strategy e id.
 * La funzione presuppone che in argv[2] ed argv[3] siano presenti dei valori.
 * @param argv array di stringhe da cui ottenere i valori da salvare in strategy e id
 * @param strategy specified strategy
 * @param id specified id
 */
void read_input(char **argv, int *strategy, int *id)
{
    *strategy = atoi(argv[2]);
    *id = atoi(argv[3]);
}

/**
 * controlla se x è potenza di due o meno .
 * Esempio : is_power_of_two(4) ritorna 1
 * @param x intero su cui effettuare il controllo
 * @return 1 se x è potenza di due , 0 altrimenti
 */
int is_power_of_two(int x)
{
    return (x & (x - 1)) == 0;
}

/**
 * Esegue i controlli dei warnings su strategia e id. Id è eventualmente modificato.
 * Esempio con strategy = 1 , id = -1 , id è posto a 0.
 * @param strategy strategy da applicare
 * @param id id specificato
 */
void warnings(int strategy, int *id)
{
    if (num_procs == 1)
    {
        printf("%s", WARNING_NPROC_ONE);
    }
    if (strategy != STRATEGY_3 && *id != 0)
    {
        printf("%s", WARNING_ID_STRATEGY);
        *id = 0;
    }
    if (strategy != STRATEGY_3 && *id == -1)
    {
        printf("%s", WARNING_ID_STRATEGY);
        *id = 0;
    }
}

/**
 * Verifica che non si ricada in un caso fatale dovuto ad un errore di input. Le variabili exit_status, strategy , num_items_input , id
 * sono passate in input e ad ogni processore sono assegnati dei valori letti ed inviati tramite MPI_Bcast.
 * I valori sono letti dalla funzione read_input() eseguita solo dal primo processo (id == 0)
 * @param memum id processo
 * @param exit_status -1 ise è presente almeno un errore
 * @param argc numero di input
 * @param strategy strategia appplicata memorizzata
 * @param argv elementi dall'input
 * @param num_items_input variabile che contiene il numero di elementi totale letto in argv[1]
 * @param id id processo che stamperò il risultato se diverso da -1 ,altrimenti stampano tutti
 */
void check_input(int memum, int *exit_status, int argc, int *strategy, char **argv, int *num_items_input, int *id)
{
    if (memum == 0)
    {
        if (argc < 4)
        {
            printf("%s", ERROR_NUM_INPUT);
            *exit_status = -1;
        }
        *num_items_input = atoi(argv[1]);
        if (argc == 4 && *num_items_input < MIN_TO_GEN_RANDOM_VALUES)
        {
            printf("%s", ERROR_RANDOM_VALUE);
            *exit_status = -1;
        }
        else
        {
            read_input(argv, strategy, id);
        }
        int num_data = argc - 4;
        if (IS_STRATEGY_OUT_OF_RANGE(*strategy))
        {
            printf("%s", ERROR_STRATEGY);
            *exit_status = -1;
        }
        else if (*strategy != STRATEGY_1 && (is_power_of_two(num_procs) != 1))
        {
            printf("%s", ERROR_NUM_PROC_STRATEGY);
            *exit_status = -1;
        }
        else if ((*id < 0 || *id > (num_procs - 1)) && *id != -1)
        {
            printf("%s", ERROR_ID_RANGE);
            *exit_status = -1;
        }
        else if (ARE_FEW_ELEMENTS(*num_items_input, num_procs))
        {
            printf("%s", ERROR_ITEMS_PROC);
            *exit_status = -1;
        }
        else if (*num_items_input != num_data && (*num_items_input < MIN_TO_GEN_RANDOM_VALUES))
        {
            printf("%s", ERROR_NUM_INPUT_WITH_ARGS);
            *exit_status = -1;
        }
        if (*exit_status != -1)
        {
            warnings(*strategy, id);
        }
    }
    MPI_Bcast(exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(strategy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(num_items_input, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

/**
 * Configura l'utilizzo della libreria MPI ed assegna , per ogni processore , il proprio memum e in num_procs il numero totale dei processori
 * @param num_elem numero input
 * @param argv tutti i dati passati input
 * @param memum id processo
 */
void init_MPI(int num_elem, char **argv, int *memum)
{
    MPI_Init(&num_elem, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, memum);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
}

/**
 * Calcola il numero di elementi che ogni processo si aspetta di ricevere
 * Esempio : (0,10,NULL,NULL), con num_procs = 2 . Allora num_items_specified = 5 , rest = 0
 * @param memum id processo
 * @param num_items_specified numero totale passato dall'input degli elementi
 * @param num_data_proc numero di elementi per il processo
 * @param rest eventuale resto che se presente e se il memum è minore del resto , il numero di elementi attesi viene maggiorato di uno.
 */
void calculate_elem_proc(int memum, int num_items_specified, int *num_data_proc, int *rest)
{
    *num_data_proc = num_items_specified / num_procs;
    *rest = num_items_specified % num_procs;
    if (memum < *rest)
    {
        *num_data_proc = *num_data_proc + 1;
    }
}

/**
 * Calcola il tempo impiegato dal processo e se memum == 0 stampa il risultato ottenuto da MPI_Reduce
 * @param start_time_proc tempo registrato da quando sono registrate le performance
 * @param memum id processo
 */
void read_performance(double start_time_proc, int memum)
{
    double time_proc = MPI_Wtime() - start_time_proc;
    double max_time;
    MPI_Reduce(&time_proc, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (memum == 0)
    {
        printf("Maximum time : %f\n", max_time);
    }
}

/**
 * memorizza nell'input un array di potenze di due pari al numero di processi
 * @param exp2 puntatore che indirizzerà ad un array di num_procs potenze di due
 */
void exponentials(int **exp2)
{
    (*exp2) = (int *)calloc(num_procs, sizeof(int));
    int multiple = 1;
    int i = 0;
    for (i = 0; i < num_procs; i++)
    {
        (*exp2)[i] = multiple;
        multiple *= 2;
    }
    return;
}

/**
 * Effettua il parsing delle stringhe fornite in input per convertirle in double oppure se num_items_specified è presente un numero maggiore di 20 ,
 * sono generati tanti double quanti num_items_specified e memorizzati in data.
 * @param argv array da formattare
 * @param memum id processo
 * @param data array in cui memorizzare gli elementi
 * @param num_items_specified numero di elementi
 */
void parse_input(char **argv, int memum, double **data, int num_items_specified)
{
    if (memum == 0)
    {
        (*data) = (double *)calloc(num_items_specified, sizeof(double));
        if (num_items_specified < MIN_TO_GEN_RANDOM_VALUES)
        {
            int i = 0;
            for (i = 4; i < num_items_specified + 4; i++)
            {
                (*data)[i - 4] = atof(argv[i]);
            }
        }
        else
        {
            srand(time(NULL));
            int i = 0;
            for (i = 0; i < num_items_specified; i++)
            {
                double gen = MIN_VALUE_GEN + (double)rand() / RAND_MAX * (MAX_VALUE_GEN - MIN_VALUE_GEN);
                // printf("generated item %f\n", gen);
                (*data)[i] = gen;
            }
        }
    }
}

/**
 * Se la funzione è eseguita con memum == 0, distribuisce gli elementi contenuti in data, ai restanti processori con un ciclo for.
 * Esempio = (0,10,0,[1,2,3,4])
 * @param memum id processo
 * @param num_data_proc elementi per processore
 * @param rest se presente , per i processi con memum < rest è passato un elemento in più
 * @param data array di elementi da distribuire
 */
void distribute_data(int memum, int num_data_proc, int rest, double *data)
{
    if (memum == 0)
    {
        int index = 0;
        int tmp = num_data_proc;
        int memum_proc = 0;
        for (memum_proc = 1; memum_proc < num_procs; memum_proc++)
        {
            index += tmp;
            if (memum_proc == rest)
            {
                tmp -= 1;
            }
            MPI_Send(&data[index], tmp, MPI_DOUBLE, memum_proc, DISTRIBUTION_TAG(memum_proc), MPI_COMM_WORLD);
        }
    }
}

/**
 * Se il memum è diverso da zero , allora il processo si mette in attesa sincrona con MPI_Recv attendendo num_data_proc elementi per memorizzarli in data
 * @param memum id processo
 * @param num_data_proc elementi attesi
 * @param data array in cui memorizzare
 */
void get_data(int memum, int num_data_proc, double **data)
{
    if (memum != 0)
    {
        (*data) = (double *)calloc(num_data_proc, sizeof(double));
        MPI_Recv(*data, num_data_proc, MPI_DOUBLE, 0, DISTRIBUTION_TAG(memum), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/**
 * memorizza in start_time il tempo ricevuto da MPI_Wtime
 * @param start_time variabile in cui memorizzare il risultato di MPI_Wtime
 */
void start_performance(double *start_time)
{
    MPI_Barrier(MPI_COMM_WORLD);
    *start_time = MPI_Wtime();
}

/**
 * Esegue una semplice somma di num_data_proc elementi sequenzialmente
 * @param data id processo
 * @param num_data_proc numero di elementi da leggere
 * @param result somma parziale del processo
 */
void local_calculation(double *data, int num_data_proc, double *result)
{
    *result = 0;
    int i = 0;
    for (i = 0; i < num_data_proc; i++)
    {
        *result += data[i];
    }
}

/**
 * Esegue la prima strategia
 * @param memum id processo
 * @param local_sum somma parziale che viene solamente letta se memum != 0, altrimenti a questa sono aggiunte le restanti somme parziali
 */
void first_strategy(int memum, double *local_sum)
{
    if (memum == 0)
    {
        double sum_proc = 0;
        int memum_proc = 0;
        for (memum_proc = 1; memum_proc < num_procs; memum_proc++)
        {
            MPI_Recv(&sum_proc, 1, MPI_DOUBLE, memum_proc, TAG_STRATEGY(memum_proc), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *local_sum += sum_proc;
        }
    }
    else
    {
        MPI_Send(local_sum, 1, MPI_DOUBLE, 0, TAG_STRATEGY(memum), MPI_COMM_WORLD);
    }
}

/**
 * Esegue la seconda strategia
 * @param memum id processo
 * @param partial_sum somma parziale che viene solamente letta se memum != 0, altrimenti a questa sono aggiunte le restanti somme parziali
 */
void second_strategy(int memum, double *partial_sum)
{
    int *exp2;
    exponentials(&exp2);
    int steps = log2(num_procs);
    double tmp_buff;
    int step = 0;
    for (step = 0; step < steps; step++)
    {
        if ((memum % exp2[step]) == 0)
        {
            if ((memum % exp2[step + 1]) == 0)
            {
                MPI_Recv(&tmp_buff, 1, MPI_DOUBLE, (memum + exp2[step]), TAG_STRATEGY(step), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                *partial_sum += tmp_buff;
            }
            else
            {
                MPI_Send(partial_sum, 1, MPI_DOUBLE, (memum - exp2[step]), TAG_STRATEGY(step), MPI_COMM_WORLD);
            }
        }
    }
}

/**
 * Esegue la terza strategia
 * @param memum id processo
 * @param partial_sum somma parziale che a fine esecuzione diviene somma totale per tutti i processi
 */
void third_strategy(int memum, double *partial_sum)
{
    int steps = 0;
    int *exp2;
    exponentials(&exp2);
    steps = log2(num_procs);
    int another_rank = 0;
    int level_multiple = 0;
    int level = 0;
    for (level = 0; level < steps; level++)
    {
        level_multiple = exp2[level];
        if ((memum % (exp2[level + 1])) < level_multiple)
        {
            another_rank = (memum + level_multiple);
            MPI_Send(partial_sum, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD);
            double buff = 0;
            MPI_Recv(&buff, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *partial_sum = *partial_sum + buff;
        }
        else
        {
            MPI_Send(partial_sum, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD);
            double buff = 0;
            MPI_Recv(&buff, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *partial_sum = *partial_sum + buff;
        }
    }
}

/**
 * In base a strategy viene deciso quale strategia applicare
 * Esempio (1,0,100)
 * @param strategy input strategia
 * @param memum id processo
 * @param local_sum somma parziale che contribuisce alla somma totale
 */
void communication_strategy(int strategy, int memum, double *local_sum)
{
    switch (strategy)
    {
    case STRATEGY_1:
        first_strategy(memum, local_sum);
        break;
    case STRATEGY_2:
        second_strategy(memum, local_sum);
        break;
    case STRATEGY_3:
        third_strategy(memum, local_sum);
        break;
    }
}

/**
 * Viene stampata la somma totale se e solo se:
 * 1. memum == id
 * 2. se id = -1 ed è indicata la terza strategia così tutti i processi stampano il risultato totale
 * @param memum id processo
 * @param id id del processo che deve stampare
 */
void print_result(int memum, int id, int strategy, double sum)
{
    if ((id == -1 && strategy == STRATEGY_3) || memum == id)
    {
        printf("Total sum : %f, printed by %d\n\n", sum, memum);
    }
}
/**
*
 @param argv vari input
*
 @param num_items numero di elementi indicato come primo parametro input
*/
void execute_seq(char **argv, int num_items)
{
    double *data;
    double result;
    double start_time = 0;
    parse_input(argv, 0, &data, num_items);
    start_performance(&start_time);
    local_calculation(data, num_items, &result);
    read_performance(start_time, 0);
    print_result(0, 0, 1, result);
}
