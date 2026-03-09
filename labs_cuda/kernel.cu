#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

cudaError_t subWithCuda(double *c, const double *a, const double *b, unsigned int size);

__global__ void subKernel(double *c, const double *a, const double *b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (index < n) {        
        c[index] = a[index] - b[index];
        index += offset;
    }
}

int main()
{
    int arraySize = 0;
    scanf("%d", &arraySize);

    double *a = (double*)malloc(arraySize * sizeof(double));
    double *b = (double*)malloc(arraySize * sizeof(double));
    double *c = (double*)malloc(arraySize * sizeof(double));

    for (int i = 0; i < arraySize; i++) {
        scanf("%lf", &a[i]);
    }

    for (int i = 0; i < arraySize; i++) {
        scanf("%lf", &b[i]);
    }

    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus  = subWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "subWithCuda failed!");
        return 1;
    }

    for (int i = 0; i < arraySize - 1; i++) {
        printf("%.10e ", c[i]);
    }
    printf("%.10e ", c[arraySize - 1]);
    return 0;
}

cudaError_t subWithCuda(double *c, const double *a, const double *b, unsigned int size)
{
    double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc1 failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc2 failed!");
        cudaFree(dev_a);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc3 failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2 failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    subKernel<<<1, 256>>>(dev_c, dev_a, dev_b, size);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
