#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <string.h>

int check_input();

int check_input(int num_proc){
    if (sqrt(num_proc) * sqrt(num_proc) != num_proc)
    {
        printf("ERRORE:Impossibile applicare la strategia...\n");
        printf("Il numero di processori deve essere tale da generare una griglia quadrata.\n\n Ad esempio 1,4,9,16,25,...\n");
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[])
{

    int menum, num_proc;
    int m, n, k; // numero di righe e colonne delle matrici
    int p, q, flag = 0, sinc = 1, offsetR = 0, offsetC = 0, dim_prod_par;
    int N, i, j, z, y, tag, mittente, mittenter, destinatarior;
    float *A, *B, *C, *subA, *subB, *subC, *Temp, *printTemp;
    double t_inizio, t_fine, T1, Tp = 0.F, speedup, Ep;
    MPI_Status info;
    MPI_Request rqst;
    MPI_Comm griglia, grigliar, grigliac;
    int coordinate[2], mittBcast[2], destRoll[2], mittRoll[2];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &menum);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    int check_input = 1;
    if(menum == 0){
        check_input(num_proc);
    }
    MPI_Bcast( &check_input , 1 , MPI_INT , 0 , MPI_COMM_WORLD);

    if(check_input == 0){
        MPI_Finalize();
        
    }

    //******************************************************
    // Inizio fase inserimento dati nelle matrici
    //******************************************************
    if (menum == 0)
    {
        printf("\n*********************************************************\n");
        printf("Algoritmo per il calcolo del prodotto Matrice per Matrice\n");
        printf("\n\nStrategia BMR\n");
        printf("\n*********************************************************\n");
        printf("\nProcessori utilizzati: %d\n", num_proc);
        // Procedura di inserimento dati nella Matrice A
        while (flag == 0)
        {
            printf("\nInserire il numero di righe della matrice A:");
            fflush(stdin);
            scanf("%d", &m);
            p = sqrt(num_proc);

            // Si richiede che il numero di righe di A sia multiplo di p
            if (m % p != 0)
                printf("ATTENZIONE:Numero di righe non divisibile per p=%d!\n", p);
            else
                flag = 1;
        }
        {
            // Si richiede che il numero di colonne di A sia multiplo di p
            printf("\nInserire il numero di colonne della matrice A:");
            fflush(stdin);
            scanf("%d", &n);
            // Si richiede che il numero di colonne di A sia multiplo di p
            if (n % p != 0)
                printf("ATTENZIONE:Numero di colonne non divisibile per p=%d!\n", p);
            else
                flag = 0;
        }
        // Procedura di inserimento dati nella Matrice A
        while (flag == 0)
        {
            printf("\nInserire il numero di colonne della matrice B:");
            fflush(stdin);
            scanf("%d", &k);
            // Si richiede che il numero di colonne di B sia multiplo di p
            if (k % p != 0)
                printf("ATTENZIONE:Numero di colonne non divisibile per p=%d!\n", p);
            else
                flag = 1;
        }
        // Allocazione dinamica delle matrici A, B e C solo in P0
        // Il numero di righe di B è pari al numero di colonne di A
        A = (float *)malloc(m * n * sizeof(float));
        B = (float *)malloc(n * k * sizeof(float));
        C = (float *)calloc(m * k, sizeof(float));
        printf("\n\n\n*****************************************************\n");
        printf("- Acquisizione elementi di A matrice %dx%d -\n", m, n);
        printf("\n*****************************************************\n");
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
            }
        }
        // Comandi per l'inserimento dei dati a mano
        // printf("Digita elemento A[%d][%d]: ", i, j);
        // scanf("%f", A+i*n+j);
        // per valutare la prestazioni si attiva la riga seguente
        *(A + i * n + j) = (float)rand() / ((float)RAND_MAX + (float)1);
        printf("\n\n\n*****************************************************\n");
        printf("- Acquisizione elementi di B matrice %dx%d -\n", n, k);
        printf("\n*****************************************************\n");
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < k; j++)
            {
                // Comandi per l'inserimento dei dati a mano
                // printf("Digita elemento B[%d][%d]: ", i, j);
                // scanf("%f", B+i*k+j);
            }
        }
        // per valutare la prestazioni si attiva la riga seguente
        *(B + i * k + j) = (float)rand() / ((float)RAND_MAX + (float)1);
        // Stampa della Matrice A solo se sono meno di 100 dati
        if (m < 100 && n < 100)
        {
            printf("\nMatrice A acquisita:");
            for (i = 0; i < m; i++)
            {
                printf("\n");
                for (j = 0; j < n; j++)
                    printf("%.2f\t", *(A + i * n + j));
            }
        } // end if
        else
        {
            printf("\nLa matrice è di grandi dimensioni \n");
            printf("\n e non verrà stampata \n");
        } // end else
        // Stampa della Matrice B solo se sono meno di 100 dati
        if (m < 100 && n < 100)
        {
            printf("\nMatrice B acquisita:");
            for (i = 0; i < n; i++)
            {
                printf("\n");
                for (j = 0; j < k; j++)
                    printf("%.2f\t", *(B + i * k + j));
            }
        } // end if
        else
        {
            printf("\nLa matrice è di grandi dimensioni \n");
            printf("\n e non verrà stampata \n");
        }
        //*************************************************************
        // Inizio fase della creazione della struttura griglia dei dati
        //*************************************************************
        printf("\n\n\n*****************************************************\n");
        printf("- Costruzione della griglia (%dx%d) di processori -", p, p);
        printf("\n*****************************************************\n");
    } // end if di riga 81
    // Il processore P0 esegue un Broadcast del numero di righe di A: m
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Il processore P0 esegue un Broadcast del numero di colonne di A: n
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Il processore P0 esegue un Broadcast del numero di colonne di B: k
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Il processore P0 esegue un broadcast del valore p
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Carica la funzione crea_griglia (dal file mxm-aux.c) per creare la
    crea_griglia(&griglia, &grigliar, &grigliac, menum, p, coordinate);
    // Tutti i processori allocano spazio per il blocco subA
    subA = (float *)malloc(m / p * n / p * sizeof(float));
    // Tutti allocano spazio per un blocco d'appoggio
    Temp = (float *)malloc(m / p * n / p * sizeof(float));
    // Tutti allocano spazio per il blocco subB
    subB = (float *)malloc(n / p * k / p * sizeof(float));
    // Tutti allocano spazio per il blocco subC
    subC = (float *)calloc(m / p * k / p, sizeof(float));
    printTemp = (float *)malloc(k / p * sizeof(float));
    if (menum == 0)
    {
        for (i = 1; i < num_proc; i++)
        {
            // Gli offset sono necessari a P0 per individuare in A il blocco da
            offsetC = m / p * (i / p); // Individua la riga da cui partire
            offsetR = n / p * (i % p); // Individua la colonna da cui partire
            for (j = 0; j < m / p; j++)
            {
                tag = 10 + i;
                // Spedisce gli elementi in vettori di dimensione n/p
                MPI_Send(A + offsetC * n + offsetR, n / p, MPI_FLOAT, i, tag,
                         MPI_COMM_WORLD);
                offsetC++; // l'offset di colonna viene aggiornato
            }              // end for
        }                  // end for
        {
            // Stessa cosa avviene per spedire sottoblocchi di B
            offsetC = n / p * (i / p);
            offsetR = k / p * (i % p);
            for (j = 0; j < n / p; j++)
            {
                tag = 20 + i;
                // Spedisce gli elementi di B in vettori di dimensione k/p
                MPI_Send(B + offsetC * k + offsetR, k / p, MPI_FLOAT, i, tag,
                         MPI_COMM_WORLD);
                offsetC++; // l'offset di colonna viene aggiornato
            }              // end for interno
        }                  // end for esterno
        // P0 inizializza il suo blocco subA
        for (j = 0; j < m / p; j++)
        {
            for (z = 0; z < n / p; z++)
                *(subA + n / p * j + z) = *(A + n * j + z);
        }
        // P0 inizializza il suo blocco subB
        for (j = 0; j < n / p; j++)
        {
            for (z = 0; z < k / p; z++)
                *(subB + k / p * j + z) = *(B + k * j + z);
        }
    } // end if di riga 252
    else
    {
        // Tutti i processori ricevono il rispettivo subA
        for (j = 0; j < m / p; j++)
        {
            tag = 10 + menum;
            MPI_Recv(subA + n / p * j, n / p, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &info);
        }
        // Ricevono subB
        for (j = 0; j < n / p; j++)
        {
            tag = 20 + menum;
            MPI_Recv(subB + k / p * j, k / p, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &info);
        }
    } // end else di riga 302
    // Tutti i processori hanno i rispettivi blocchi subA e subB
    for (i = 0; i < p; i++)
    {
        // al primo passo fa broadcast e prodotto
        if (i == 0)
        {
            // Calcola le coordinate del processore che invierà in broadcast
            // la propria matrice subA ai processori sulla stessa riga
            mittBcast[0] = coordinate[0];
            mittBcast[1] = (i + coordinate[0]) % p;
            // Calcola le coordinate del processore a cui inviare subB
            destRoll[0] = (coordinate[0] + p - 1) % p;
            destRoll[1] = coordinate[1];
            // Calcola le coordinate del processore da cui ricevere la nuova subB
            mittRoll[0] = (coordinate[0] + 1) % p;
            mittRoll[1] = coordinate[1];
            // Ricava il rango del destinatario sulla propria colonna
            MPI_Cart_rank(grigliac, destRoll, &destinatarior);
            // Ricava il rango del mittente sulla propria colonna
            MPI_Cart_rank(grigliac, mittRoll, &mittenter);
            // Ricava il rango del processore che effettua il Broadcast sulla
            MPI_Cart_rank(grigliar, mittBcast, &mittente);
            if (coordinate[0] == mittBcast[0] && coordinate[1] == mittBcast[1])
            {
                // L'esecutore del broadcast copia subA nel blocco temporaneo
            }
            memcpy(Temp, subA, n / p * m / p * sizeof(float));
            t_inizio = MPI_Wtime();
            MPI_Bcast(Temp, n / p * m / p, MPI_FLOAT, mittente, grigliar);
            t_fine = MPI_Wtime();
            Tp += t_fine - t_inizio;
            Tp += mat_mat_righe(Temp, m / p, n / p, subB, k / p, subC);
        }    // end if di riga 324
        else // Se non è il primo passo esegue Broadcast, Rolling e Prodotto
        {
            mittBcast[0] = coordinate[0];
            mittBcast[1] = (i + coordinate[0]) % p;
            if (coordinate[0] == mittBcast[0] && coordinate[1] == mittBcast[1])
            {
                // L'esecutore del broadcast copia subA nel blocco temporaneo
                memcpy(Temp, subA, n / p * m / p * sizeof(float));
            }
            t_inizio = MPI_Wtime();
            // Broadcast di Temp
            MPI_Bcast(Temp, n / p * m / p, MPI_FLOAT, mittente, grigliar);

            // Il rolling vede l'invio del blocco subB al processore della riga
            tag = 30; // La spedizione è non bloccante mentre la ricezione si
            MPI_Isend(subB, n / p * k / p, MPI_FLOAT, destinatarior, tag, grigliac, &rqst);
            // E la ricezione del nuovo blocco subB dalla riga inferiore
            tag = 30;
            MPI_Recv(subB, n / p * k / p, MPI_FLOAT, mittenter, tag, grigliac, &info);
            t_fine = MPI_Wtime();
            // Calcola il prodotto parziale e il tempo impiegato per eseguirlo
            Tp += mat_mat_righe(Temp, m / p, n / p, subB, k / p, subC) + t_fine - t_inizio;
        } // end else di riga 364
    }     // end for di riga 321
    // Tutti i processori in ordine inviano a P0 la propria porzione di C
    if (menum == 0)
    {
        // P0 stampa così come le riceve le porzioni ricevute
        printf("\n*********************************************************\n");
        printf("* Stampa del risultato e dei parametri di valutazione *\n");
        printf("\n*********************************************************\n");
        printf("Matrice risultato:\n");
        if (m < 100 && n < 100)
        {
            for (i = 0; i < p; i++)
            { // per quante sono le righe di C
                for (z = 0; z < m / p; z++)
                { // per quante sono le righe di subC
                    for (j = 0; j < p; j++)
                    { // per quanti sono i processori per riga
                        if (i * p + j != 0)
                        {
                            tag = 70;
                            MPI_Recv(printTemp, k / p, MPI_FLOAT, i * p + j, tag, MPI_COMM_WORLD, &info);
                            for (y = 0; y < k / p; y++)
                                // Stampa la porzione di riga di C ricevuta
                                printf(" %.2f\t", *(printTemp + y));
                        } // end if
                        else
                            // P0 stampa le righe della propria matrice subC
                            for (y = 0; y < k / p; y++)
                                printf(" %.2f\t", *(subC + k / p * z + y));
                    } // end for riga 414
                    printf("\n");
                } // end for riga 412
            }     // end for riga 410
        }         // end if riga 408
        else
        {
            printf("\nLa matrice è di grandi dimensioni \n");
            printf("\n e non verrà stampata \n");
        } // end else
        // Calcolo del prodotto con singolo processore e del tempo necessario
            T1 = mat_mat_righe(A, m, n, B, k, C);
        speedup = T1 / Tp;    // Calcola lo speed up
        Ep = speedup / num_proc; // Calcola l'efficienza
        // Stampa dei risultati ottenuti
        printf("\n\nIl tempo di esecuzione su un processore e' %f secondi\n",T1);
        printf("Il tempo di esecuzione su %d processori e' %f secondi\n", num_proc, Tp);
        printf("Lo Speed Up ottenuto e' %f\n", speedup);
        printf("L'efficienza risulta essere %f\n\n", Ep);

        free(A);
        free(B);
    }
    else
    {
        for (i = 0; i < m / p; i++)
        {
            tag = 70;
            MPI_Send(subC + i * k / p, k / p, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
        }
    }
    free(Temp);
    free(printTemp);
    free(subA);
    free(subB);
    MPI_Finalize();
    return (0);
}
