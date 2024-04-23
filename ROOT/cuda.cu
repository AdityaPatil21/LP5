#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul(int* A, int* B, int* C, int N) {
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    if (Row < N && Col < N) {
        int Pvalue = 0;
        for (int k = 0; k < N; k++) {
            Pvalue += A[Row*N+k] * B[k*N+Col];
        }
        C[Row*N+Col] = Pvalue;
    }
}

int main() {
    int N = 512;
    int size = N * N * sizeof(int);
    int* A, * B, * C;
    int* dev_A, * dev_B, * dev_C;
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N+j] = i*N+j;
            B[i*N+j] = j*N+i;
        }
    }

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);

    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}

// program 2 

#include <iostream>  
#include <cuda_runtime.h>

using namespace std;

__global__ void addVectors(int* A, int* B, int* C, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) 
    {
        C[i] = A[i] + B[i];
    }
}

int main() 
{
    int n = 1000000;  
    int* A, * B, * C;
    int size = n * sizeof(int);

    // Allocate memory on the host  
    cudaMallocHost(&A, size);  
    cudaMallocHost(&B, size);  
    cudaMallocHost(&C, size);

    // Initialize the vectors
    for (int i = 0; i < n; i++) 
    {
        A[i] = i;
        B[i] = i * 2;
    }
    // Allocate memory on the device  
    int* dev_A, * dev_B, * dev_C;  
    cudaMalloc(&dev_A, size);  
    cudaMalloc(&dev_B, size);  
    cudaMalloc(&dev_C, size);

    // Copy data from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);  
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Launch the kernel  
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    // Copy data from device to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < 10; i++) 
    {
        cout << C[i] << " ";
    }
    cout << endl;

    // Free memory  
    cudaFree(dev_A);  
    cudaFree(dev_B);  
    cudaFree(dev_C);  
    cudaFreeHost(A);  
    cudaFreeHost(B);  
    cudaFreeHost(C);

    return 0;
}



















These two programs demonstrate the use of CUDA (Compute Unified Device Architecture) for parallel computing on GPUs (Graphics Processing Units).

Program 1: Matrix Multiplication

The matmul kernel function is defined to perform matrix multiplication of two square matrices A and B to obtain matrix C.
Inside the kernel, each thread calculates a single element of the resulting matrix C by iterating through the rows of A and columns of B.
The main function initializes matrices A and B on the CPU, and then copies them to the GPU memory.
Grid and block dimensions (dimGrid and dimBlock) are configured for launching the kernel, where each block contains 16x16 threads.
The matmul kernel is launched with the specified grid and block dimensions to perform matrix multiplication on the GPU.
The resulting matrix C is copied back to the CPU memory and printed to the console.
Program 2: Vector Addition

The addVectors kernel function is defined to perform element-wise addition of two vectors A and B to obtain vector C.
Inside the kernel, each thread calculates a single element of the resulting vector C.
The main function initializes vectors A and B on the CPU, and then copies them to the GPU memory.
Grid and block dimensions are configured for launching the kernel, where each block contains 256 threads.
The addVectors kernel is launched with the specified grid and block dimensions to perform vector addition on the GPU.
The resulting vector C is copied back to the CPU memory and printed to the console.
Both programs demonstrate the steps involved in using CUDA for parallel computation on GPUs, including memory allocation, data transfer between CPU and GPU, kernel launch configuration, kernel execution, and result retrieval.









CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It enables developers to utilize the computational power of NVIDIA GPUs (Graphics Processing Units) for general-purpose processing tasks, in addition to their traditional graphics rendering capabilities.

CUDA provides a programming environment and a set of tools for developing parallel applications that run on NVIDIA GPUs. It includes a C/C++ compiler, runtime libraries, development tools, and APIs that allow developers to write parallel code that executes on the GPU.

Key features of CUDA include:

Parallelism: CUDA enables developers to harness the parallel processing capabilities of GPUs to accelerate computation-intensive tasks. It allows for massive parallelism by dividing tasks into smaller threads that can be executed concurrently on GPU cores.
CUDA C/C++ Programming: CUDA extends the C/C++ programming language with GPU-specific extensions, allowing developers to write code that runs both on the CPU and GPU. CUDA provides constructs for managing parallel threads, memory management, and synchronization.
CUDA Runtime API: CUDA provides a runtime API that allows developers to manage devices, allocate memory, launch kernels (parallel functions executed on the GPU), and synchronize execution between the CPU and GPU.
CUDA Libraries: NVIDIA provides a collection of optimized libraries for common parallel computing tasks, such as linear algebra, signal processing, and image processing. These libraries leverage the parallel processing power of GPUs to accelerate computation.
CUDA Toolkit: The CUDA Toolkit includes development tools, profilers, debuggers, and other utilities for building, optimizing, and debugging CUDA applications. It provides a comprehensive development environment for GPU-accelerated computing.
Overall, CUDA has become a widely used platform for accelerating a wide range of scientific, engineering, and data processing applications by leveraging the computational power of GPUs.
