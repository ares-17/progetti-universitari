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
#define ARE_FEW_ELEMENTS(num_items, num_proc)(num_items / 2) < num_proc
#define IS_STRATEGY_OUT_OF_RANGE(strategy) strategy < 1 || strategy > 3
#define ERROR_NUM_INPUT "Errore : il programma necessita di 3 input per calcolare la somma di numeri casuali\n"\
"oppure almeno 5 per il calcolo della somma degli elementi in input:\n"\
"\tsomma numeri casuali :\n"\
"\t\t ^1 Input come numero di elementi da generare e sommare (maggiore di 20)\n"\
"\t\t ^2 Input indica la strategia da applicare (compreso tra 1 e 3)\n"\
"\t\t ^3 Input indica il processo che stampa il risultato (indicare -1 indicare una stampa da tutti i processi)\n"\
"\t somma numeri in input:\n"\
"\t\t ^1 Input come numero di elementi da generare e sommare (minore di 21)\n"\
"\t\t ^2 Input indica la strategia da applicare (compreso tra 1 e 3)\n"\
"\t\t ^3 Input indica il processo che stampa il risultato (indicare -1 indicare una stampa da tutti i processi)\n"\
"\t\t [^4, ...] Input che indicano gli elementi da sommare\n\n"
#define ERROR_RANDOM_VALUE "Se intendi generare gli elementi della somma in maniera casuale, occorrono gli input:\n"\
"1^ ^1 Input come numero di elementi da generare e sommare (minore di 21)\n"\
"^2 Input indica la strategia da applicare (compreso tra 1 e 3)\n"\
"^3 Input indica il processo che stampa il risultato (indicare -1 indicare una stampa da tutti i processi)\n\n"
#define ERROR_STRATEGY "La strategia deve esser compresa tra 1 e 3 ...\n"
#define WARNING_ID_STRATEGY "Se e' selezionata la prima o la seconda strategia allora, solo il primo processo (id = 0) ha il risultato totale!\n"\
"Per tale motivo l'id e' posto a 0...\n\n"
#define ERROR_ITEMS_PROC "Specificare almeno due elementi per processo!\n"
#define ERROR_NUM_PROC_STRATEGY "Le strategie 2 e 3 sono applicabili su un numero di processi che e' potenza di 2\n"
#define ERROR_ID_RANGE "L'id del processo deve esser compreso tra 0 e il (numero di processi - 1),!\n"\
" oppure indicare -1 per far stampare il calcolo totale da tutti i processi\n\n"
#define WARNING_NPROC_ONE "Specificando un singolo processo, la libreria MPI non e' utilizzata\n"\
"e l'intero calcolo e' eseguito sequenzialmente ...\n\n"
#define ERROR_NUM_INPUT_WITH_ARGS "Il numero totale di elementi da sommare non corrisponde agli elementi effettivamente passati in input \n"
#define DISTRIBUTION_TAG(i) 501 + i
#define MIN_TO_GEN_RANDOM_VALUES 21
#define MAX_VALUE_GEN 100
#define MIN_VALUE_GEN 1
#define EXIT_STATUS_ERROR - 1

int const MIN_ARG_REQUIRED = 6;
int num_procs;

int check_num_items_input(int * num_items_input, int argc);
int check_id(int * id);
int check_strategy(int * strategy);
void distribuite_data(int memum, double * data, int num_data_proc, double ** recv_buffer);
void parse_input(char ** argv, int memum, int num_data_proc, int num_total_items, double ** recv_buffer);

void read_input(char ** argv, int * strategy, int * id);
int is_power_of_two(int x);
void warnings(int * strategy, int * id);
void check_input(int memum, int * exit_status, int argc, int * strategy, char ** argv, int * num_items_input, int * id);
void init_MPI(int num_elem, char ** data, int * memum);
int calculate_elem_proc(int memum, int num_items_specified, int * rest);
void read_performance(double start_time_proc, int memum);
void exponentials(int ** exp2);
void start_performance(double * start_time);
void local_calculation(double * data, int num_data_proc, double * result);
void first_strategy(int memum, double * local_sum);
void second_strategy(int memum, double * partial_sum);
void third_strategy(int memum, double * partial_sum);
void communication_strategy(int strategy, int memum, double * local_sum);
void print_result(int memum, int id, int strategy, double sum);
void execute_seq(char ** argv, int num_items);

int main(int argc, char ** argv) {
  int memum;
  int num_data_proc = 0;
  int rest;
  double * recv_buffer;
  double local_sum;
  double start_time;
  int exit_status = 0;
  int strategy = 0;
  int num_items_input = 0;
  int id = 0;

  init_MPI(argc, argv, & memum);
  check_input(memum, & exit_status, argc, & strategy, argv, & num_items_input, & id);

  if (exit_status != 0) {
    MPI_Finalize();
    exit(exit_status);
  } else if (num_procs == 1) {
    execute_seq(argv, num_items_input);
    MPI_Finalize();
    return 0;
  }
  num_data_proc = calculate_elem_proc(memum, num_items_input, & rest);
  parse_input(argv, memum, num_data_proc, num_items_input, & recv_buffer);

  start_performance( & start_time);
  local_calculation(recv_buffer, num_data_proc, & local_sum);
  communication_strategy(strategy, memum, & local_sum);
  print_result(memum, id, strategy, local_sum);
  read_performance(start_time, memum);

  if (recv_buffer != NULL) {
    free(recv_buffer);
  }

  MPI_Finalize();
}

/**
 * Distribuisce send_buffer ai processi in MPI_COMM_WORLD utilizzando MPI_Gather e MPI_Scatterv :
 *  MPI_Gather per assegnare ad un array di memum = 0 il numero di elementi che ad ogni processo spetta
 *  MPI_Scatterv per distribuire un numero variabile di elementi ai processi , secondo i dati ricavati da MPI_Gather
 * @param memum
 * @param send_buffer la totalita' degli elementi dell'operazione
 * @param num_data_proc numero di elementi che ciascun processo avra' dopo la funzione
 * @param recv_buffer array in cui memorizzare gli elementi del processo
*/
void distribuite_data(int memum, double * send_buffer, int num_data_proc, double ** recv_buffer) {
  int * items_for_process = (memum == 0) ? (int * ) calloc(num_procs, sizeof(int)) : NULL;
  int * displacements = (memum == 0) ? (int * ) calloc(num_procs, sizeof(int)) : NULL;
  ( * recv_buffer) = (double * ) calloc(num_data_proc, sizeof(double));

  MPI_Gather( & num_data_proc, 1, MPI_INT, items_for_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (memum == 0) {
    int i = 0;
    displacements[0] = 0;
    for (i = 1; i < num_procs; i++) {
      displacements[i] = displacements[i - 1] + items_for_process[i - 1];
    }
  }
  MPI_Scatterv(send_buffer, items_for_process, displacements, MPI_DOUBLE, * recv_buffer, num_data_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (items_for_process != NULL) {
    free(items_for_process);
  }

  if (displacements != NULL) {
    free(displacements);
  }
}

/**
 * controlla se x è potenza di due o meno .
 * @example : is_power_of_two(4) ritorna 1
 * @param x intero su cui effettuare il controllo
 * @return 1 se x è potenza di due , 0 altrimenti
 */
int is_power_of_two(int x) {
  return (x & (x - 1)) == 0;
}

/**
 * Esegue i controlli dei warnings su strategia e id. Id è eventualmente modificato.
 * @example strategy = 1 , id = -1 , id è posto a 0.
 * @param strategy strategy da applicare
 * @param id id specificato
 */
void warnings(int * strategy, int * id) {
  if (num_procs == 1) {
    printf("%s", WARNING_NPROC_ONE);
  }
  if ( * strategy != STRATEGY_3 && ( * id != 0 || * id == -1)) {
    printf("%s", WARNING_ID_STRATEGY);
    * id = 0;
  }
}

/**
 * Verifica che non si ricada in un caso fatale dovuto ad un errore di input. Le variabili exit_status, strategy , num_items_input , id
 * sono passate in input e ad ogni processo sono assegnati dei valori letti ed inviati tramite MPI_Bcast.
 * Il processo con memum=0 effettua i controlli
 * @param memum id processo
 * @param exit_status -1 ise è presente almeno un errore
 * @param argc numero di input
 * @param strategy strategia appplicata memorizzata
 * @param argv elementi dall'input
 * @param num_items_input variabile che contiene il numero di elementi totale letto in argv[1]
 * @param id id processo che stamperò il risultato se diverso da -1 ,altrimenti stampano tutti
 */
void check_input(int memum, int * exit_status, int argc, int * strategy, char ** argv, int * num_items_input, int * id) {

  if (memum == 0) {
    if (argc < 4) {
      printf("%s", ERROR_NUM_INPUT);
      * exit_status = EXIT_STATUS_ERROR;
    } else {
      * num_items_input = atoi(argv[1]);
      * strategy = atoi(argv[2]);
      * id = atoi(argv[3]);
      * exit_status =
        check_num_items_input(num_items_input, argc) &&
        check_id(id) &&
        check_strategy(strategy) ?
        0 :
        EXIT_STATUS_ERROR;
      warnings(strategy, id);
    }
  }
  MPI_Bcast(exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(id, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(strategy, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(num_items_input, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

/**
 * Esegue controlli di qualita' sul primo input che indica il numero di elementi
 * @param num_items_input numero di elementi input
 * @param argc numero di elementi totali in input al programma  
*/
int check_num_items_input(int * num_items_input, int argc) {
  int items_input = argc - 4;

  if ( * num_items_input < 0) {
    printf("%s", "Indicato un numero negativo di elementi da sommare pari a :%d\n", * num_items_input);
    return 0;
  } else if ( * num_items_input != items_input && ( * num_items_input < MIN_TO_GEN_RANDOM_VALUES)) {
    printf("%s", ERROR_NUM_INPUT_WITH_ARGS);
    return 0;
  } else if (ARE_FEW_ELEMENTS( * num_items_input, num_procs)) {
    printf("%s", ERROR_ITEMS_PROC);
    return 0;
  }
  return 1;
}

/**
 * Esegue controlli sul valore del terzo parametro , l'id del processo
 * @param id id processo
*/
int check_id(int * id) {
  if (( * id < -1) || ( * id >= num_procs)) {
    printf("%s", ERROR_ID_RANGE);
    return 0;
  }
  return 1;
}

/**
 * Verifica che la strategia sia coerente con il numero di processi e controllo che il valore ricada nel range
 * @param strategy strategia
*/
int check_strategy(int * strategy) {
  if (IS_STRATEGY_OUT_OF_RANGE( * strategy)) {
    printf("%s", ERROR_STRATEGY);
    return 0;
  } else if ( * strategy != STRATEGY_1 && (is_power_of_two(num_procs) != 1)) {
    printf("%s", "Strategia impostata sul valore di 1 poiché il numero di processi non è potenza di 2...\n");
    * strategy = STRATEGY_1;
  }
  return 1;
}

/**
 * Configura l'utilizzo della libreria MPI ed assegna per ogni processo , il proprio memum e in num_procs il numero totale dei processi
 * @param num_elem numero input
 * @param argv tutti i dati passati input
 * @param memum id processo
 */
void init_MPI(int num_elem, char ** argv, int * memum) {
  MPI_Init( & num_elem, & argv);
  MPI_Comm_rank(MPI_COMM_WORLD, memum);
  MPI_Comm_size(MPI_COMM_WORLD, & num_procs);
}

/**
 * Calcola il numero di elementi che ogni processo si aspetta di ricevere
 * @example : (0,10,NULL), con num_procs = 2 . Allora num_items_specified = 5 , rest = 0
 * @param memum id processo
 * @param num_items_specified numero totale passato dall'input degli elementi
 * @param rest eventuale resto che se presente e se il memum è minore del resto , il numero di elementi attesi viene maggiorato di uno.
 */
int calculate_elem_proc(int memum, int num_items_specified, int * rest) {
  int num_data_proc = 0;
  num_data_proc = num_items_specified / num_procs;
  * rest = num_items_specified % num_procs;
  if (memum < * rest) {
    num_data_proc = num_data_proc + 1;
  }
  return num_data_proc;
}

/**
 * Calcola il tempo impiegato dal processo e se memum == 0 stampa il risultato ottenuto da MPI_Reduce
 * @param start_time_proc tempo registrato da quando sono registrate le performance
 * @param memum id processo
 */
void read_performance(double start_time_proc, int memum) {
  double time_proc = MPI_Wtime() - start_time_proc;
  double max_time;
  MPI_Reduce( & time_proc, & max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (memum == 0) {
    printf("Maximum time : %f\n", max_time);
  }
}

/**
 * memorizza nell'input un array di potenze di due pari al numero di processi
 * @param exp2 puntatore che indirizzerà ad un array di num_procs potenze di due
 */
void exponentials(int ** exp2) {
  ( * exp2) = (int * ) calloc(num_procs, sizeof(int));
  int multiple = 1;
  int i = 0;
  for (i = 0; i < num_procs; i++) {
    ( * exp2)[i] = multiple;
    multiple *= 2;
  }
  return;
}

/**
 * Se gli elementi sono letti dall'input , il processo memum=0 ne esegue la conversione in double , poi saranno distribuiti.
 * Se gli elementi devono essere generati, ogni processo genera autonomamente i suoi elementi.  
 * @param argv array da formattare
 * @param memum id processo
 * @param num_data_proc numero di elementi che spettano al processo
 * @param num_total_items numero di elementi totali
 * @param recv_buffer array in cui memorizzare gli elementi del processo
 */
void parse_input(char ** argv, int memum, int num_data_proc, int num_total_items, double ** recv_buffer) {
  if (num_total_items < MIN_TO_GEN_RANDOM_VALUES) {
    double * send_buffer = NULL;
    if (memum == 0) {
      send_buffer = (double * ) calloc(num_total_items, sizeof(double));
      int i = 0;
      for (i = 4; i < num_total_items + 4; i++) {
        send_buffer[i - 4] = atof(argv[i]);
      }
    }
    distribuite_data(memum, send_buffer, num_data_proc, recv_buffer);
  } else {
    ( * recv_buffer) = (double * ) calloc(num_data_proc, sizeof(double));
    int i = 0;
    srand(time(NULL));
    for (i = 0; i < num_data_proc; i++) {
      double gen = MIN_VALUE_GEN + (double) rand() / RAND_MAX * (MAX_VALUE_GEN - MIN_VALUE_GEN);
      ( * recv_buffer)[i] = gen;
    }
  }
}

/**
 * memorizza in start_time il tempo ricevuto da MPI_Wtime
 * @param start_time variabile in cui memorizzare il risultato di MPI_Wtime
 */
void start_performance(double * start_time) {
  MPI_Barrier(MPI_COMM_WORLD);
  * start_time = MPI_Wtime();
}

/**
 * Esegue una semplice somma di num_data_proc elementi sequenzialmente
 * @param data id processo
 * @param num_data_proc numero di elementi da leggere
 * @param result somma parziale del processo
 */
void local_calculation(double * data, int num_data_proc, double * result) {
  * result = 0;
  int i = 0;
  for (i = 0; i < num_data_proc; i++) {
    * result += data[i];
  }
}

/**
 * Esegue la prima strategia
 * @param memum id processo
 * @param local_sum somma parziale che viene solamente letta se memum != 0, altrimenti a questa sono aggiunte le restanti somme parziali
 */
void first_strategy(int memum, double * local_sum) {
  if (memum == 0) {
    double sum_proc = 0;
    int memum_proc = 0;
    for (memum_proc = 1; memum_proc < num_procs; memum_proc++) {
      MPI_Recv( & sum_proc, 1, MPI_DOUBLE, memum_proc, TAG_STRATEGY(memum_proc), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      * local_sum += sum_proc;
    }
  } else {
    MPI_Send(local_sum, 1, MPI_DOUBLE, 0, TAG_STRATEGY(memum), MPI_COMM_WORLD);
  }
}

/**
 * Esegue la seconda strategia
 * @param memum id processo
 * @param partial_sum somma parziale che viene solamente letta se memum != 0, altrimenti a questa sono aggiunte le restanti somme parziali
 */
void second_strategy(int memum, double * partial_sum) {
  int * exp2;
  exponentials( & exp2);
  int steps = log2(num_procs);
  double tmp_buff;
  int step = 0;
  for (step = 0; step < steps; step++) {
    if ((memum % exp2[step]) == 0) {
      if ((memum % exp2[step + 1]) == 0) {
        MPI_Recv( & tmp_buff, 1, MPI_DOUBLE, (memum + exp2[step]), TAG_STRATEGY(step), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        * partial_sum += tmp_buff;
      } else {
        MPI_Send(partial_sum, 1, MPI_DOUBLE, (memum - exp2[step]), TAG_STRATEGY(step), MPI_COMM_WORLD);
      }
    }
  }
  if (exp2 != NULL) {
    free(exp2);
  }
}

/**
 * Esegue la terza strategia
 * @param memum id processo
 * @param partial_sum somma parziale che a fine esecuzione diviene somma totale per tutti i processi
 */
void third_strategy(int memum, double * partial_sum) {
  int steps = 0;
  int * exp2;
  exponentials( & exp2);
  steps = log2(num_procs);
  int another_rank = 0;
  int level_multiple = 0;
  int level = 0;
  for (level = 0; level < steps; level++) {
    level_multiple = exp2[level];
    if ((memum % (exp2[level + 1])) < level_multiple) {
      another_rank = (memum + level_multiple);
      MPI_Send(partial_sum, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD);
      double buff = 0;
      MPI_Recv( & buff, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      * partial_sum = * partial_sum + buff;
    } else {
      another_rank = (memum - level_multiple);
      MPI_Send(partial_sum, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD);
      double buff = 0;
      MPI_Recv( & buff, 1, MPI_DOUBLE, another_rank, TAG_STRATEGY(level), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      * partial_sum = * partial_sum + buff;
    }
  }
  if (exp2 != NULL) {
    free(exp2);
  }
}

/**
 * In base a strategy viene deciso quale strategia applicare
 * @example : (1,0,100)
 * @param strategy input strategia
 * @param memum id processo
 * @param local_sum somma parziale che contribuisce alla somma totale
 */
void communication_strategy(int strategy, int memum, double * local_sum) {
  switch (strategy) {
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
 * 1. memum == id in input
 * 2. se id = -1 ed è indicata la terza strategia così tutti i processi stampano il risultato totale
 * @param memum id processo
 * @param id id del processo che deve stampare
 */
void print_result(int memum, int id, int strategy, double sum) {
  if ((id == -1 && strategy == STRATEGY_3) || memum == id) {
    printf("Total sum : %f, da id = %d\n\n", sum, memum);
  }
}

/**
 * @param argv vari input
 * @param num_items numero di elementi indicato come primo parametro input
 */
void execute_seq(char ** argv, int num_items) {
  double * recv_buffer;
  double result;
  double start_time = 0;

  parse_input(argv, 0, num_items, num_items, & recv_buffer);
  start_performance( & start_time);
  local_calculation(recv_buffer, num_items, & result);
  read_performance(start_time, 0);
  print_result(0, 0, 1, result);
}