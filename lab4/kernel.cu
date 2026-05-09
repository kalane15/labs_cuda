%% writefile kernel.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

const double EPS = 1e-7;

struct AbsCompare {
    __host__ __device__
        bool operator()(double a, double b) const {
        return fabs(a) < fabs(b);
    }
};

__global__ void swap_rows(double* A, int n, int r1, int r2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    double tmp = A[col * n + r1];
    A[col * n + r1] = A[col * n + r2];
    A[col * n + r2] = tmp;
}

__global__ void eliminate2D_opt(double* A, int n, int k, double inv) {
    __shared__ double shared_row[32];
    __shared__ double shared_col[32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.x * blockDim.x + tx + k + 1;
    int col = blockIdx.y * blockDim.y + ty + k + 1;

    if (tx == 0 && col < n)
        shared_row[ty] = A[col * n + k];

    if (ty == 0 && row < n)
        shared_col[tx] = A[k * n + row];

    __syncthreads();

    if (row >= n || col >= n) return;

    double Aik = shared_col[tx];
    double Akj = shared_row[ty];

    A[col * n + row] -= (Aik * inv) * Akj;
}

double determinant(std::vector<double>& A, int n) {
    double* d_A;
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    int swaps = 0;

    for (int k = 0; k < n; ++k) {
        thrust::device_ptr<double> col_ptr(d_A + k * n + k);

        auto max_it = thrust::max_element(
            col_ptr,
            col_ptr + (n - k),
            AbsCompare()
        );

        int pivot = (max_it - thrust::device_pointer_cast(d_A)) % n;

        double Akk;
        cudaMemcpy(&Akk, d_A + k * n + pivot,
            sizeof(double), cudaMemcpyDeviceToHost);

        if (fabs(Akk) < EPS) {
            cudaFree(d_A);
            return 0.0;
        }

        if (pivot != k) {
            swaps++;
            swap_rows << <(n + 255) / 256, 256 >> > (d_A, n, k, pivot);
        }

        cudaMemcpy(&Akk, d_A + k * n + k,
            sizeof(double), cudaMemcpyDeviceToHost);

        double inv = 1.0 / Akk;

        dim3 block(32, 32);
        dim3 grid(
            (n - k + block.x - 1) / block.x,
            (n - k + block.y - 1) / block.y
        );

        eliminate2D_opt << <grid, block >> > (d_A, n, k, inv);
    }

    std::vector<double> h_A(n * n);
    cudaMemcpy(h_A.data(), d_A, n * n * sizeof(double),
        cudaMemcpyDeviceToHost);

    cudaFree(d_A);

    double det = 1.0;
    for (int i = 0; i < n; ++i)
        det *= h_A[i * n + i];

    if (swaps % 2) det = -det;

    return det;
}

int main() {
    int n;
    std::cin >> n;

    std::vector<double> A(n * n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            std::cin >> A[j * n + i];

    double det = determinant(A, n);
    printf("%.10e\n", det);

    return 0;
}