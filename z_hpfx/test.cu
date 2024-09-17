#include <iostream>
#include <cuda_runtime.h>
#include<cudnn.h>

__global__ void hello() {
    printf("Hello, CUDA!\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}