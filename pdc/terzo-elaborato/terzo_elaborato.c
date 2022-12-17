#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

typedef struct {
    int menum;
    int menum_cart;
    int coordinate[2];
    int num_procs;
    int num_procs_dim_cart;
    double* buffer_first_matrix;
    double* buffer_second_matrix;
    int num_data_dim;
    int num_total_items;
} LocalData;

int read_input(int argc, char **argv, int *length);
int init_MPI(int num_elem, char **argv, int *menum);
double **gen_square_matrix(int length);
void print_matrix(double **matrix, int length);
int create_square_cart_procs(int dim_length, LocalData *data, MPI_Comm *comm_cart);
void create_process_cart(LocalData *data, MPI_Comm* comm_grid);
int check_is_square(int dim_length, int *result);
int *get_period();
double *distribuite_matrix_on_proc_cart(LocalData *data, double **matrix, int dim_matrix);
void distribuite_data(LocalData *data, double** matrix, int current_col, int current_row, int dim_matrix);
void print_local_data(LocalData data);
void MBR(LocalData data,double** first_matrix, int dim_matrix, MPI_Comm comm_grid);
void share_items_first_matrix(LocalData data, MPI_Comm comm_grid, int dim_matrix);
void share_items_second_matrix(LocalData data, MPI_Comm comm_grid);


int const NUMERO_DIM = 2;

int main(int argc, char **argv)
{
    LocalData *data = malloc(sizeof(LocalData));
    int dim_length, check_input;
    double **first_matrix, **second_matrix;
    MPI_Comm comm_cart;

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

    int result_cart_create = create_square_cart_procs(dim_length, data, &comm_cart);

    if(result_cart_create == 0){
        MPI_Finalize();
        exit(-1);
    }

    if(data->menum == 0){
        first_matrix = gen_square_matrix(dim_length);
        second_matrix = gen_square_matrix(dim_length);
        print_matrix(first_matrix, dim_length);
    }
	MPI_Barrier(MPI_COMM_WORLD);

    data->buffer_first_matrix = distribuite_matrix_on_proc_cart(data, first_matrix, dim_length);
    data->buffer_second_matrix = distribuite_matrix_on_proc_cart(data, second_matrix, dim_length);

    if(data->menum == 0){
        print_local_data(*data);
        
    }
    MBR(*data,first_matrix,dim_length,comm_cart);

    MPI_Finalize();
}

int init_MPI(int num_elem, char **argv, int *menum)
{
	MPI_Init(&num_elem, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, menum);
    int num_procs;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    return num_procs;
}

int read_input(int argc, char **argv, int *length){
    if(argc != 2 || (atoi(argv[1]) < 0)){
        printf("Numero di elementi errato\n");
        return 0;
    }
    *length = atoi(argv[1]);
    return 1;
}

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

int create_square_cart_procs(int dim_length, LocalData *data, MPI_Comm *comm_cart){
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

    create_process_cart(data, comm_cart);
    return 1;
}

void create_process_cart(LocalData *data, MPI_Comm* comm_grid){
    int num_elementi_per_dimensione[2] = {data->num_procs_dim_cart, data->num_procs_dim_cart};

    MPI_Cart_create(MPI_COMM_WORLD, NUMERO_DIM, num_elementi_per_dimensione, get_period(), 0, comm_grid);
    MPI_Comm_rank(*comm_grid, &data->menum_cart);
    MPI_Cart_coords(*comm_grid, data->menum_cart, NUMERO_DIM, data->coordinate);
}

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

double *distribuite_matrix_on_proc_cart(LocalData *data, double **matrix, int dim_matrix){
    double* buffer = NULL;

    if (data->menum == 0){        
        data->num_data_dim = (dim_matrix / data->num_procs_dim_cart);
        data->num_total_items = (data->num_data_dim * data->num_data_dim);
        buffer = (double*)calloc((data->num_total_items),sizeof(double));

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
    MPI_Bcast(&data->num_total_items , 1, MPI_INT , 0, MPI_COMM_WORLD);

    if(data->menum != 0){
        buffer = (double*)calloc((data->num_total_items),sizeof(double));
        MPI_Recv(buffer , (data->num_total_items), MPI_DOUBLE ,0 , 77 + data->menum , MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return buffer;
}

void distribuite_data(LocalData *data, double** matrix, int current_col, int current_row, int dim_matrix){
    int menum_slave = 1;
    MPI_Request request;

    for(menum_slave = 1; menum_slave < data->num_procs; menum_slave++){
        int index_buffer = 0;
        double* buffer = (double*)calloc((data->num_total_items),sizeof(double));
        int i = 0;
        for (i = current_row; i < (current_row + data->num_data_dim); i++){
            int j = current_col;
            for (j = current_col; j < (current_col + data->num_data_dim); j++){
                buffer[index_buffer] = matrix[i][j];
                index_buffer++;
            } 
        }
        MPI_Isend(buffer ,data->num_total_items,MPI_DOUBLE , menum_slave , 77 + menum_slave , MPI_COMM_WORLD , &request);
        
        current_col += data->num_data_dim;
        if(current_col >= dim_matrix){
            current_col = 0;
            current_row += data->num_data_dim;
        }

        free(buffer);
    }
}

void MBR(LocalData data, double** first_matrix, int dim_matrix, MPI_Comm comm_grid){
    /*
    int row = 0, col = 0;
    int entry;
    for(entry = 0; entry < dim_matrix; entry++){
        row = 0;
        int num_elem ;
        col = entry;
        for(num_elem = 0; num_elem < dim_matrix; num_elem++){
            //printf("[%d,%d]:%.1f\t",row, col,first_matrix[row][col]);
            share_items_first_matrix(data, comm_grid);
            share_items_second_matrix(data, comm_grid);
            row++;
            col++;
            if(col == dim_matrix){
                col = 0;
            }
        }
        printf("\n");
    }
    */

   if(data.menum == 4){
        share_items_first_matrix(data, comm_grid,dim_matrix);
        share_items_second_matrix(data, comm_grid);
   }
}

void share_items_first_matrix(LocalData data, MPI_Comm comm_grid, int dim_matrix){
    MPI_Request request;
    int col = 0;
    for(col = 0; col< data.num_procs_dim_cart; col++){
        if( col != data.coordinate[1]){
            int menum_dest, coords_dest[] = {data.coordinate[0], col};
            MPI_Cart_rank(comm_grid, coords_dest,&menum_dest); 
            MPI_Isend(data.buffer_second_matrix , 
                data.num_total_items, 
                MPI_DOUBLE, 
                menum_dest , 
                11 + menum_dest , 
                comm_grid , 
                &request 
            );
            double* buffer = (double*)calloc((data.num_total_items),sizeof(double));
            MPI_Irecv(&buffer , 
                data.num_total_items , 
                MPI_DOUBLE , 
                menum_dest , 
                11 + menum_dest , 
                comm_grid , 
                &request);
            printf("mando a %d,%d \t",coords_dest[0],coords_dest[1]);
        }
    }
    printf("\n");
}

void share_items_second_matrix(LocalData data, MPI_Comm comm_grid){
    MPI_Request request;

    int dest_row = ((data.coordinate[0] - 1) < 0) ? 
        (data.num_procs_dim_cart - 1): 
        (data.coordinate[0] - 1);
    int menum_dest , coords_dest[2] = {dest_row, data.coordinate[1]};
    MPI_Cart_rank(comm_grid, coords_dest,&menum_dest); 
    MPI_Isend(data.buffer_second_matrix , 
        data.num_total_items, 
        MPI_DOUBLE, 
        menum_dest , 
        11 + menum_dest , 
        comm_grid , 
        &request);

    int source_row = ((data.coordinate[0] + 1) == data.num_procs_dim_cart) ? 0 : (data.coordinate[0] + 1);
    int source, coords_source[] = {source_row, data.coordinate[1]};
    double* buffer = (double*)calloc((data.num_total_items),sizeof(double));
    MPI_Cart_rank(comm_grid, coords_source,&source); 
    MPI_Irecv(&buffer , 
        data.num_total_items , 
        MPI_DOUBLE , 
        source , 
        11 + source , 
        comm_grid , 
        &request);

    printf("\nmando [%d,%d], ricevo da [%d,%d]\n",coords_dest[0], coords_dest[1], coords_source[0], coords_source[1]);
}

void print_local_data(LocalData data){
    printf("menum: %d\n \
        menum_cart: %d\n \
        coordinate: [%d,%d]\n \
        num_procs: %d\n \
        num_procs_dim_cart: %d\n \
        num_data_dim: %d\n \
        num_total_items: %d\n",
        data.menum,
        data.menum_cart,
        data.coordinate[0],
        data.coordinate[1],
        data.num_procs,
        data.num_procs_dim_cart,
        data.num_data_dim,
        data.num_total_items);
    printf("first_buffer:[");
    for(int i = 0; i < data.num_total_items; i++){
        printf("%.2f ,",data.buffer_first_matrix[i]);
    }
    printf("]\n");
    printf("second_buffer:[");
    for(int i = 0; i < data.num_total_items; i++){
        printf("%.2f ,",data.buffer_second_matrix[i]);
    }
    printf("]\n");
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