#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <string.h>

typedef struct {
    int menum;
    int menum_cart;
    int coordinate[2];
    int num_procs;
    int num_procs_dim_cart;
    double* buffer_first_matrix;
    double* buffer_second_matrix;
    double* buffer_third_matrix;
    int num_data_dim;
    int num_items_first;
    int num_items_second;
    int num_items_third;
} LocalData;


void start_performance(double *start_time);
int read_input(int argc, char **argv, int *length);
int init_MPI(int num_elem, char **argv, int *menum);
double **gen_square_matrix(int length);
void print_matrix(double **matrix, int length);
int create_square_cart_procs(int dim_length, LocalData *data, MPI_Comm *comm_cart, MPI_Comm* grigliar, MPI_Comm* grigliac);
void create_process_cart(LocalData *data, MPI_Comm* comm_grid, MPI_Comm* grigliar, MPI_Comm* grigliac);
int check_is_square(int dim_length, int *result);
int *get_period();
double *distribuite_matrix_on_proc_cart(LocalData *data, double **matrix, int dim_matrix);
void distribuite_data(LocalData *data, double** matrix, int current_col, int current_row, int dim_matrix);
void MBR(LocalData* data, MPI_Comm grigliar, MPI_Comm grigliac);
void print_array(double* tmp, int total_items);
double* all_gather_rows(MPI_Comm comm, LocalData* data );
double* all_gather_columns(MPI_Comm comm, LocalData* data );
double* prod_matrix_as_array(double* matrix_a, double* matrix_b, int num_elem_dim, int num_elem_proc);
double* sum_matrix_as_array(double* matrix_a, double* matrix_b, int num_elem_dim, int num_elem_proc);
void read_performance(double start_time_proc, int memum);

int const NUMERO_DIM = 2;

int main(int argc, char **argv)
{
    LocalData *data = malloc(sizeof(LocalData));
    int dim_length, check_input;
    double **first_matrix, **second_matrix, start_time;
    MPI_Comm comm_cart, grigliac, grigliar;

    data->num_procs = init_MPI(argc, argv, &data->menum);

    if(data->menum == 0){
        check_input = read_input(argc, argv, &dim_length);
    }
    MPI_Bcast(&dim_length ,1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast(&check_input , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    if(check_input == 0){
        MPI_Finalize();
        exit(-1);
    }

    int result_cart_create = create_square_cart_procs(dim_length, data, &comm_cart, &grigliar, &grigliac);

    if(result_cart_create == 0){
        MPI_Finalize();
        exit(-1);
    }

    if(data->menum == 0){
        first_matrix = gen_square_matrix(dim_length);
        second_matrix = gen_square_matrix(dim_length);
        if((dim_length * dim_length) <= 81){
            print_matrix(first_matrix, dim_length);
        }
    }
	MPI_Barrier(MPI_COMM_WORLD);

    data->buffer_first_matrix = distribuite_matrix_on_proc_cart(data, first_matrix, dim_length);
    data->buffer_second_matrix = distribuite_matrix_on_proc_cart(data, second_matrix, dim_length);

    start_performance(&start_time);
    MBR(data,grigliar,grigliac);
    read_performance(start_time,data->menum);

    MPI_Finalize();
}

/**
 * inizializzazione dell'ambiente MPI
*/
int init_MPI(int num_elem, char **argv, int *menum)
{
	MPI_Init(&num_elem, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, menum);
    int num_procs;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    return num_procs;
}

/**
 * Legge gli input e se non presenti ritorna True
*/
int read_input(int argc, char **argv, int *length){
    if(argc != 2 || (atoi(argv[1]) < 0)){
        printf("Numero di elementi errato\n");
        return 0;
    }
    *length = atoi(argv[1]);
    return 1;
}

/**
 * genera una matrice quadrata con valori random
 * @param length dimensioni della matrice
*/
double **gen_square_matrix(int length){
    double** matrix = (double**)calloc(length,sizeof(double*));
    int i = 0;
    int index = 1;
    for(i = 0; i < length; i++){
        matrix[i] = (double*)calloc(length,sizeof(double));
        int j = 0;
        for (j = 0; j < length; j++){
            matrix[i][j] = index;
            index++;
        }
    }
    return matrix;
}

/**
 * Applica delle verifiche preliminari prima di creare una griglia di processori quadrata. La comunicazione degli esiti
 * delle verifiche e' impiegata con MPI_Bcast
 * @param dim_length lunghezza dimensione della matrice
 * @param data valori locali del processo
 * @param comm_cart communicator comune ai processi
 * @param grigliar communicator di sole righe
 * @param grigliac communicator di sole colonne
*/
int create_square_cart_procs(int dim_length, LocalData *data, MPI_Comm *comm_cart, MPI_Comm* grigliar, MPI_Comm* grigliac){
    int is_dim_multiple_nprocs, is_nprocs_square;
    if(data->menum == 0){
        is_dim_multiple_nprocs = (dim_length % data->num_procs) == 0;
        is_nprocs_square = check_is_square(data->num_procs, &data->num_procs_dim_cart);
    }
    MPI_Bcast(&is_dim_multiple_nprocs, 1, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&is_nprocs_square, 1, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&data->num_procs_dim_cart, 1, MPI_INT, 0 , MPI_COMM_WORLD);

    if(is_dim_multiple_nprocs == 0){
        if(data->menum == 0){
            printf("il numero di elementi per dimensione non è multiplo del numero di processi\n");     
        }
        return 0;
    }
    if( is_nprocs_square == 0){
        if(data->menum == 0){
            printf("il numero di processi non è un quadrato\n");
        }
        return 0;
    }

    create_process_cart(data, comm_cart, grigliar, grigliac);
    return 1;
}

/**
 * @param data valori locali del processo
 * @param comm_cart communicator comune ai processi
 * @param grigliar communicator di sole righe
 * @param grigliac communicator di sole colonne
*/
void create_process_cart(LocalData *data, MPI_Comm* comm_grid, MPI_Comm* grigliar, MPI_Comm* grigliac){
    int num_elementi_per_dimensione[2] = {data->num_procs_dim_cart, data->num_procs_dim_cart};

    MPI_Cart_create(MPI_COMM_WORLD, NUMERO_DIM, num_elementi_per_dimensione, get_period(), 0, comm_grid);
    MPI_Comm_rank(*comm_grid, &data->menum_cart);
    MPI_Cart_coords(*comm_grid, data->menum_cart, NUMERO_DIM, data->coordinate);

    MPI_Cart_sub(*comm_grid, (int[2]){0, 1}, grigliac);
    MPI_Cart_sub(*comm_grid, (int[2]){1,0}, grigliar);
}

/**
 * Verifica se il parametro in ingresso e' un quadrato. 
 * in caso affermativo il puntatore result punta ad un valore non NULL e il valore 0 e' ritornato
*/
int check_is_square(int dim_length, int *result){
    double tmp = sqrt(dim_length); 
    int is_square = tmp == (int)tmp;
    *result = (int)tmp;
    return is_square;
}

int *get_period() {
    int *tmp = (int *)calloc(NUMERO_DIM, sizeof(int));
    tmp[0] = tmp[1] = 0;
    return tmp;
}

/**
 * Distribuisce la matrice in sotto-matrici di ugual dimensione per ogni processo
 * @param data variabili locali del processo
 * @param matrix matrice da distribuire
 * @param dim_matrix dimensione della matrice
*/
double *distribuite_matrix_on_proc_cart(LocalData *data, double **matrix, int dim_matrix){
    double* buffer = NULL;

    if (data->menum == 0){        
        data->num_data_dim = (dim_matrix / data->num_procs_dim_cart);
        data->num_items_first = (data->num_data_dim * data->num_data_dim);
        data->num_items_second = data->num_items_first; 
        buffer = (double*)calloc((data->num_items_first),sizeof(double));

        int current_row = 0;
        int current_col = 0;
        int index_buffer = 0;
        int i = 0;

        for (i = current_row; i < (current_row + data->num_data_dim); i++){
            int j = current_col;
            for (j = current_col; j < (current_col + data->num_data_dim); j++){
                buffer[index_buffer] = matrix[i][j];
                index_buffer++;
            } 
        }
        current_col += data->num_data_dim;
        
        distribuite_data(data, matrix, current_col, current_row, dim_matrix);
    }
    MPI_Bcast(&data->num_data_dim , 1, MPI_INT , 0, MPI_COMM_WORLD);
    MPI_Bcast(&data->num_items_first , 1, MPI_INT , 0, MPI_COMM_WORLD);
    MPI_Bcast(&data->num_items_second,1,MPI_INT,0,MPI_COMM_WORLD);

    if(data->menum != 0){
        buffer = (double*)calloc((data->num_items_first),sizeof(double));
        MPI_Recv(buffer , (data->num_items_first), MPI_DOUBLE ,0 , 77 + data->menum , MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return buffer;
}

/**
 * Distribuisce con MPI_Isend le sottomatrici ai restanti processi
 * @param data variabili locali del processo
 * @param matrix matrice da distribuire
 * @param dim_matrix dimensione della matrice
*/
void distribuite_data(LocalData *data, double** matrix, int current_col, int current_row, int dim_matrix){
    int menum_slave = 1;
    MPI_Request request;

    for(menum_slave = 1; menum_slave < data->num_procs; menum_slave++){
        int index_buffer = 0;
        double* buffer = (double*)calloc((data->num_items_first),sizeof(double));
        int i = 0;
        for (i = current_row; i < (current_row + data->num_data_dim); i++){
            int j = current_col;
            for (j = current_col; j < (current_col + data->num_data_dim); j++){
                buffer[index_buffer] = matrix[i][j];
                index_buffer++;
            } 
        }
        MPI_Isend(buffer ,data->num_items_first,MPI_DOUBLE , menum_slave , 77 + menum_slave , MPI_COMM_WORLD , &request);
        
        current_col += data->num_data_dim;
        if(current_col >= dim_matrix){
            current_col = 0;
            current_row += data->num_data_dim;
        }

        free(buffer);
    }
}

/**
 * Applica l'algoritmo MBR in tal modo:
 *  Ogni processo riceve dagli altri della stessa riga le sottomatrici corrispondenti di A
 *  Ogni processo riceve dagli altri della stessa colonna le sottomatrici corrispondenti di B
 *  Esegue i prodotti a coppie di matrici secondo l'algoritmo MBR
 *  Esegue la somma dei prodotti ricavati
 * @param data valori locali del processo
 * @param grigliar communicator di sole righe
 * @param grigliac communicator di sole colonne
*/
void MBR(LocalData* data, MPI_Comm grigliar, MPI_Comm grigliac){
    int i = 0, j = 0, z = 0;

    double* all_elem_row = all_gather_rows(grigliac, data);
    double* all_elem_col = all_gather_columns(grigliar, data);

    const int all_size = data->num_items_first * data->num_procs_dim_cart;
    const int num_elem_proc = (all_size / data->num_procs_dim_cart);
    const int num_elem_dim = sqrt(num_elem_proc);

    data->num_items_third = num_elem_proc;
    data->buffer_third_matrix = (double*)calloc(num_elem_proc, sizeof(double));

    double* third_matrix = (double*)calloc(all_size, sizeof(double));

    // prodotto tra ogni sotto-matrice di A e della corrispondente sotto-matrice in B sfruttando l'ordinamento
    // ricavato da MPI_Allgather 
    int matrix_index = 0;
    for(matrix_index = 0; matrix_index < all_size; matrix_index += num_elem_proc){
        double* matrix_a = (all_elem_row + matrix_index);
        double* matrix_b = (all_elem_col + matrix_index);
        double* result = prod_matrix_as_array(matrix_a,matrix_b, num_elem_dim, num_elem_proc);
        memmove((third_matrix + matrix_index), result, sizeof(double) * num_elem_proc);
    }

    // Somma dei prodotti risultanti, in un'unica sotto-matrice C
    double* sum_result = third_matrix;
    for(matrix_index = num_elem_proc; matrix_index < all_size; matrix_index += num_elem_proc){
        double* matrix_b = (third_matrix + matrix_index);
        sum_result = sum_matrix_as_array(sum_result,matrix_b, num_elem_dim, num_elem_proc);
    }

    memmove(data->buffer_third_matrix,sum_result, sizeof(double) * num_elem_proc);
    free(third_matrix);
}

/**
 * Legge due matrici, come array, e ne applica il prodotto
 * @param matrix_a prima matrice
 * @param matrix_b seconda matrice
 * @param num_elem_dim numero di elementi su una singola dimensione delle matrici
 * @param num_elem_proc numero totale di valori della matrice 
*/
double* prod_matrix_as_array(double* matrix_a, double* matrix_b, int num_elem_dim, int num_elem_proc){
    int i , j , z;
    double* result = (double*)calloc(num_elem_proc, sizeof(double));

    for(i = 0; i < num_elem_dim; i++){
        for(j = 0; j < num_elem_dim; j++){
            double res = 0;
            for(z = 0; z < num_elem_dim; z++){
                res += matrix_a[(i * num_elem_dim) + z] * matrix_b[(num_elem_dim * z) + j];
            }
            result[(num_elem_dim * i) + j] = res;
        }
    }
    return result;
}

/**
 * Legge due matrici, come array, e ne applica la somma
 * @param matrix_a prima matrice
 * @param matrix_b seconda matrice
 * @param num_elem_dim numero di elementi su una singola dimensione delle matrici
 * @param num_elem_proc numero totale di valori della matrice 
*/
double* sum_matrix_as_array(double* matrix_a, double* matrix_b, int num_elem_dim, int num_elem_proc){
    int i , j;
    double* result = (double*)calloc(num_elem_proc, sizeof(double));

    for(i = 0; i < num_elem_dim; i++){
        for(j = 0; j < num_elem_dim; j++){
            const int offset = (num_elem_dim * i )+ j; 
            result[offset] = matrix_a[offset] + matrix_b[offset];
        }
    }
    return result;
}

/**
 * Algoritmo che ritorna in output un array che corrisponde alla concatenazione di tutte le sottomatrici
 * della riga corrispondente
 * @param comm communicator
 * @param data variabili locali processo
*/
double* all_gather_rows(MPI_Comm comm, LocalData* data ){
    const int total_items = (*data).num_items_first * (*data).num_procs_dim_cart;
    double* recv_buffer = (double*)calloc(total_items, sizeof(double));

    MPI_Allgather((*data).buffer_first_matrix , (*data).num_items_first ,MPI_DOUBLE , recv_buffer ,
        (*data).num_items_first , MPI_DOUBLE , comm);

    return recv_buffer;
}

/**
 * Algoritmo che ritorna in output un array che corrisponde alla concatenazione di tutte le sottomatrici
 * della colonna corrispondente
 * @param comm communicator
 * @param data variabili locali processo
*/
double* all_gather_columns(MPI_Comm comm, LocalData* data ){
    const int total_items = (*data).num_items_second * (*data).num_procs_dim_cart;
    double* recv_buffer = (double*)calloc(total_items, sizeof(double));
    MPI_Allgather((*data).buffer_second_matrix , (*data).num_items_second ,MPI_DOUBLE , recv_buffer ,
        (*data).num_items_second , MPI_DOUBLE , comm);

    return recv_buffer;
}

void print_array(double* tmp, int total_items){
    int num_row = 0;
    for(num_row = 0; num_row < total_items; num_row++){
        printf("%.2f\t",tmp[num_row]);
    }
    printf("\n");
}

void print_matrix(double **matrix, int length)
{
    int i, j;
    printf("\nMATRIX\n");
    for (i = 0; i < length; i++)
    {
        printf("| ");
        for (j = 0; j < length; j++)
        {
            printf( "%5.3f | ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * 	memorizza in start_time il tempo ricevuto da MPI_Wtime
 * 	@param start_time  variabile in cui memorizzare il risultato di MPI_Wtime
 */
void start_performance(double *start_time)
{
	MPI_Barrier(MPI_COMM_WORLD);
	*start_time = MPI_Wtime();
}

/**
 * Calcola il tempo impiegato dal processo e se memum == 0 stampa il risultato ottenuto da MPI_Reduce
 * @param start_time_proc  tempo registrato da quando sono registrate le performance
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
