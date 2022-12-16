#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "mpi.h"

void leggi_righe(int menum, int argc, char **argv, int *row);
int *getPeriod();
double** makeRandomMatrix();
int is_prime(int n,int** divisors,int* length);
void calc_two_divisors(int* divisors,int length,int target,int *first,int *second);
void print(double* array,int num);
void reorder_num_proc_matrix(int *num_proc_row,int* num_proc_col);
void print_matrix(double** matrix);
double* partial_row_sums(int num_row, int num_col, double* local_buffer, double* vector);
double* make_random_vector();
void distribuite_data(int nproc , int index_buffer,int step_col, int step_row,double** matrix,double* vector, int current_col, int current_row);
void create_process_cart(int num_proc_row, int num_proc_col,int* menum_grid, MPI_Comm* comm_grid);
void create_process_cart_only_cols(MPI_Comm* comm_rows, MPI_Comm comm_grid,int* menum_rows);

int const NUMERO_DIM = 2;
int const NUM_COLS = 8;
int const NUM_ROWS = 16;

MPI_Request request;


int main(int argc, char **argv)
{
    int menum, nproc, row, col, menum_grid;
    int step_col,step_row,*divisors,num_divisors,nproc_is_prime, num_proc_col, num_proc_row, menum_rows;
    double **matrix, *local_buffer,*local_vector, *vector;

    MPI_Comm comm_grid;
    MPI_Comm comm_rows;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &menum);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if(menum == 0){
        nproc_is_prime = is_prime(nproc,&divisors,&num_divisors);
        calc_two_divisors(divisors,num_divisors,nproc,&num_proc_row,&num_proc_col);
        reorder_num_proc_matrix(&num_proc_row,&num_proc_col);
        
    }
    MPI_Bcast(&nproc_is_prime ,1 ,MPI_INT ,0 ,MPI_COMM_WORLD);
    MPI_Bcast(&num_proc_row ,1 ,MPI_INT ,0 ,MPI_COMM_WORLD);
    MPI_Bcast(&num_proc_col ,1 ,MPI_INT ,0 ,MPI_COMM_WORLD);

    if(nproc_is_prime == 1){
        if(menum == 0){
            printf("il numero processori è un numero primo! Impossibile dividere i processori in una griglia!\n");
        }
        MPI_Finalize();
        exit(-1);
    }

    if (menum == 0){
        matrix = makeRandomMatrix();
        vector = make_random_vector();
        
        print_matrix(matrix);
        step_col = NUM_COLS / num_proc_col;
        int resto_col = NUM_COLS % num_proc_col;
        
        step_row = NUM_ROWS / num_proc_row;
        int resto_row = NUM_ROWS % num_proc_row;

        int current_row = 0;
        int current_col = 0;
        int index_buffer = 0;
        int i = 0;
        local_buffer = (double*)calloc((step_col * step_row),sizeof(double));
        memcpy(local_vector ,vector, sizeof(local_vector) * step_row);
        for (i = current_row; i < (current_row + step_row); i++){
            int j = current_col;
            for (j = current_col; j < (current_col + step_col); j++){
                local_buffer[index_buffer] = matrix[i][j];
                index_buffer++;
            } 
        }
        current_col += step_col;
        
        distribuite_data(nproc, index_buffer, step_col, step_row, matrix, vector, current_col, current_row);
    }else{
        step_col = NUM_COLS / num_proc_col;
        step_row = NUM_ROWS / num_proc_row;
        local_buffer =  (double*)calloc((step_col * step_row),sizeof(double));
        local_vector = (double*)calloc((NUM_ROWS / num_proc_row),sizeof(double));
        MPI_Recv(local_vector , (NUM_ROWS / num_proc_row), MPI_DOUBLE ,0 , 77 + menum , MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(local_buffer , (step_col * step_row), MPI_DOUBLE ,0 , 77 + menum , MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    create_process_cart(num_proc_row,num_proc_col,&menum_grid,&comm_grid);
    create_process_cart_only_cols(&comm_rows,comm_grid,&menum_rows);

    double* partial_sums = partial_row_sums(step_row, step_col, local_buffer,local_vector);
    double* total_sums = (double*)calloc(step_row,sizeof(double));
    MPI_Allreduce( partial_sums , total_sums , step_row , MPI_DOUBLE , MPI_SUM , comm_rows);

    if(menum == 1){
        print(total_sums,step_row);
    }

    MPI_Finalize();
    return 0;
}

void create_process_cart(int num_proc_row, int num_proc_col,int* menum_grid, MPI_Comm* comm_grid){
    int* coordinate = (int *)calloc(NUMERO_DIM, sizeof(int));
    int* num_elementi_per_dimensione = (int *)calloc(NUMERO_DIM, sizeof(int));
    num_elementi_per_dimensione[0] = num_proc_row;
    num_elementi_per_dimensione[1] = num_proc_col;
    int* period = getPeriod();
    int reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD, NUMERO_DIM, num_elementi_per_dimensione, period, reorder, comm_grid);
    MPI_Comm_rank(*comm_grid, menum_grid);
    MPI_Cart_coords(*comm_grid, *menum_grid, NUMERO_DIM, coordinate);
}

void create_process_cart_only_cols(MPI_Comm* comm_rows, MPI_Comm comm_grid,int* menum_rows){
    int dims_cart_sub[] = {0,1}; 
    int *coordinate_rows = (int *)calloc(NUMERO_DIM, sizeof(int));
    MPI_Cart_sub(comm_grid, dims_cart_sub, comm_rows);
    MPI_Comm_rank(*comm_rows, menum_rows);
    MPI_Cart_coords(*comm_rows, *menum_rows, NUMERO_DIM - 1, coordinate_rows);
}

void distribuite_data(int nproc , int index_buffer,int step_col, int step_row,double** matrix,double* vector, int current_col, int current_row){
    int menum_slave = 1;
    for(menum_slave = 1; menum_slave < nproc; menum_slave++){
        index_buffer = 0;
        double *buffer = (double*)calloc((step_col * step_row),sizeof(double));
        int i = 0;
        for (i = current_row; i < (current_row + step_row); i++){
            int j = current_col;
            for (j = current_col; j < (current_col + step_col); j++){
                buffer[index_buffer] = matrix[i][j];
                index_buffer++;
            } 
        }
        MPI_Isend((vector + current_row), step_row, MPI_DOUBLE, menum_slave,77 + menum_slave , MPI_COMM_WORLD , &request);
        MPI_Isend(buffer ,(step_col * step_row),MPI_DOUBLE , menum_slave , 77 + menum_slave , MPI_COMM_WORLD , &request);
        
        current_col += step_col;
        if(current_col >= NUM_COLS){
            current_col = 0;
            current_row += step_row;
        }

        free(buffer);
    }
}

int is_prime(int n,int** divisors,int* length){
    *length = 0;
    int index = 0;
    if (n == 0 || n == 1)   return 0;
    if ( n % 2 == 0){
        *divisors = (int*)calloc(1,sizeof(int));
        *divisors[index] = n /2;
        *length = 1;
        return 0;
    }
    (*divisors) = (int*)calloc((n/2)+1,sizeof(int));

    for (int i = 2; i <= n / 2; ++i){
        if (n % i == 0){
            (*divisors)[index] = i;
            index++;
        }
    }
    *length = index;
    return *length == 0;
}

void calc_two_divisors(int* divisors,int length,int target,int *first,int *second){
    if(length == 1){
        *first = *second = divisors[0];
        return;
    }
    int i = 0;
    for(i = 1; i < length; i++){
        if(divisors[i] == 0)    break;
        if(divisors[i-1] * divisors[i] == target){
            *first = divisors[i-1];
            *second = divisors[i];
            return;
        }else if(divisors[i-1] * divisors[i-1] == target){
            *first = *second = divisors[i-1];
            return;
        }
    }
}

void reorder_num_proc_matrix(int *num_proc_row,int* num_proc_col){
    if (NUM_COLS > NUM_ROWS && *num_proc_row > *num_proc_col){
        int tmp = *num_proc_row;
        *num_proc_row = *num_proc_col;
        *num_proc_col = tmp;
    } else if( NUM_ROWS > NUM_COLS && *num_proc_col > *num_proc_row){
        int tmp = *num_proc_row;
        *num_proc_row = *num_proc_col;
        *num_proc_col = tmp;
    }
}

double* partial_row_sums(int num_row, int num_col, double* local_buffer, double* vector){
    double *sums = (double*)calloc(num_row,sizeof(double));
    int tmp = 0, count = 0;
    for(int i = 0; i < (num_col * num_row); i++){
        sums[tmp] += (local_buffer[i] * vector[tmp]);
        count++;
        if(count == num_col){
            count = 0;
            tmp++;
        }
    }
    
    return sums;
}

void print(double* array,int num){
    printf("ARRAY ---\n");
    for(int i  = 0; i < num; i++){
        printf("%.2f\t",array[i]);
    }
    printf("\n");
}

void print_matrix(double** matrix){
    printf("MATRIX ---\n");
    for(int i  = 0; i < NUM_ROWS; i++){
        for(int j = 0;j < NUM_COLS; j++){
            printf("%.2f\t",matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double** makeRandomMatrix(){
    double** matrix = (double**)calloc(NUM_ROWS,sizeof(double*));
    int i = 0;
    int tmp = 0;
    for(i = 0; i < NUM_ROWS; i++){
        matrix[i] = (double*)calloc(NUM_COLS,sizeof(double));
        int j = 0;
        for(j = 0; j < NUM_COLS; j++){
            matrix[i][j] = j + i;
            tmp++;
        }
    }
    return matrix;
}

double* make_random_vector(){
    double* vector = (double*)calloc(NUM_ROWS,sizeof(double));
    for(int i = 0; i < NUM_ROWS; i++){
        vector[i] = 2;
    }
    print(vector,NUM_ROWS);
    return vector;
}

int *getPeriod()
{
    int *tmp = (int *)calloc(NUMERO_DIM, sizeof(int));
    tmp[0] = tmp[1] = 0;
    return tmp;
}
