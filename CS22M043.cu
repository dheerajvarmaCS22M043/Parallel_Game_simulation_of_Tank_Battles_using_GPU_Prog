#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void initializeHP(int* gHp, int T, int H, int* gScore){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    gHp[id] = H;
    gScore[id] = 0;
}

__global__ void intitializeMini(int* gMini, int* gHit, int* gsignal){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    gMini[id] = INT_MAX;
    gHit[id] = 1001;

    if(id == 0) gsignal[0] = 0;
}


__global__ void computeHits(int* gHp, int* gScore, int* gXcoord, int* gYcoord, int round, int T, int* gMini, int* gHit){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(gHp[id] <= 0) return;
    int target = (id + round) % T;
    for(int i = 0; i < T; i++){
        if(i == id) continue;
        if(gHp[i] <= 0) continue;

        if(gXcoord[id] == gXcoord[target]){
            if(gXcoord[id] == gXcoord[i]){
                if(gYcoord[id] < gYcoord[target]){
                    if(gYcoord[i] > gYcoord[id] and abs(gYcoord[i] - gYcoord[id]) < gMini[id]){
                        gMini[id] = abs(gYcoord[i] - gYcoord[id]);
                        gHit[id] = i;
                    }
                }
                else{
                    if(gYcoord[i] < gYcoord[id] and abs(gYcoord[i] - gYcoord[id]) < gMini[id]){
                        gMini[id] = abs(gYcoord[i] - gYcoord[id]);
                        gHit[id] = i;
                    }
                }
            }
        }
        else if(gYcoord[id] == gYcoord[target]){
            if(gYcoord[id] == gYcoord[i]){
                if(gXcoord[id] < gXcoord[target]){
                    if(gXcoord[i] > gXcoord[id] and abs(gXcoord[i] - gXcoord[id]) < gMini[id]){
                        gMini[id] = abs(gXcoord[i] - gXcoord[id]);
                        gHit[id] = i;
                    }
                }
                else{
                    if(gXcoord[i] < gXcoord[id] and abs(gXcoord[i] - gXcoord[id]) < gMini[id]){
                        gMini[id] = abs(gXcoord[i] - gXcoord[id]);
                        gHit[id] = i;
                    }
                }
            }
        }
        else{
            if((gYcoord[target] - gYcoord[id]) * (gXcoord[i] - gXcoord[id]) == (gYcoord[i] - gYcoord[id]) * (gXcoord[target] - gXcoord[id])){
                if(gXcoord[id] < gXcoord[target]){
                    if(gXcoord[i] > gXcoord[id] and abs(gXcoord[i] - gXcoord[id]) < gMini[id]){
                        gMini[id] = abs(gXcoord[i] - gXcoord[id]);
                        gHit[id] = i;
                    }
                }
                else{
                    if(gXcoord[i] < gXcoord[id] and abs(gXcoord[i] - gXcoord[id]) < gMini[id]){
                        gMini[id] = abs(gXcoord[i] - gXcoord[id]);
                        gHit[id] = i;
                    }
                }
            }
        }
    }
}


__global__ void updateHP(int* gHp, int* gScore, int* gMini, int* gHit, int* gsignal){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(gMini[id] != INT_MAX){
        atomicAdd(&gScore[id], 1);
        atomicAdd(&gHp[gHit[id]], -1);
    }

    __syncthreads();
    if(gHp[id] > 0){
        atomicAdd(&gsignal[0], 1);
    }
}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *gHp; //gpu arrays
    int *gScore;
    int *gXcoord;
    int *gYcoord;

    cudaMalloc(&gHp, sizeof(int)*(T));
    cudaMalloc(&gScore, sizeof(int)*(T));
    cudaMalloc(&gXcoord, sizeof(int)*(T));
    cudaMalloc(&gYcoord, sizeof(int)*(T));

    int *gMini;
    int *gHit;

    cudaMalloc(&gMini, sizeof(int)*(T));
    cudaMalloc(&gHit, sizeof(int)*(T));

    cudaMemcpy(gScore, score, sizeof(int)*(T), cudaMemcpyHostToDevice);

    initializeHP<<<1,T>>>(gHp, T, H, gScore);
    cudaDeviceSynchronize();


    
    cudaMemcpy(gXcoord, xcoord, sizeof(int)*(T), cudaMemcpyHostToDevice);
    cudaMemcpy(gYcoord, ycoord, sizeof(int)*(T), cudaMemcpyHostToDevice);

    int* gsignal;
    cudaMalloc(&gsignal, sizeof(int));

    int round = 1;
    while(1){
        if(round % T == 0){
            round++;
            continue;
        }

        intitializeMini<<<1,T>>>(gMini, gHit, gsignal);
        cudaDeviceSynchronize();
        computeHits<<<1, T>>>(gHp, gScore, gXcoord, gYcoord, round, T, gMini, gHit);
        cudaDeviceSynchronize();
        
        updateHP<<<1, T>>>(gHp, gScore, gMini, gHit, gsignal);
        cudaDeviceSynchronize();

        int* hsignal;
        hsignal = (int*)malloc(sizeof (int));
        cudaMemcpy(hsignal, gsignal, sizeof(int), cudaMemcpyDeviceToHost);
        
        if(hsignal[0] <= 1)
            break;
        free(hsignal);

        ++round; //last line of the loop
    }

    cudaMemcpy(score, gScore, sizeof(int) * (T), cudaMemcpyDeviceToHost);

    cudaFree(gHp);
    cudaFree(gScore);
    cudaFree(gXcoord);
    cudaFree(gYcoord);
    cudaFree(gMini);
    cudaFree(gHit);
    cudaFree(gsignal);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}