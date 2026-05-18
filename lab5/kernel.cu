#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static constexpr int BLOCK = 256;
static constexpr int LOCAL = 1024;
static constexpr int INF = 0x7fffffff;

__device__ __forceinline__ void swapDir(int& a, int& b, int dir) {
    int mn = min(a, b);
    int mx = max(a, b);
    a = dir ? mn : mx;
    b = dir ? mx : mn;
}

__global__ void oddEvenShared(int* data, int n) {
    __shared__ int s[LOCAL + 1];

    int tid = threadIdx.x;
    int base = blockIdx.x << 10;
    int gid = base + tid;

    s[tid] = (gid < n) ? data[gid] : INF;

    __syncthreads();

    int lim = n - base;
    if (lim > LOCAL) lim = LOCAL;

    for (int phase = 0; phase < lim; ++phase) {
        int idx = (tid << 1) + (phase & 1);
        if (idx + 1 < lim) {
            int a = s[idx];
            int b = s[idx + 1];
            int mn = min(a, b);
            int mx = max(a, b);
            s[idx] = mn;
            s[idx + 1] = mx;
        }
        __syncthreads();
    }

    if (blockIdx.x & 1) {
        int rid = lim - tid - 1;
        if (tid < (lim >> 1)) {
            int tmp = s[tid];
            s[tid] = s[rid];
            s[rid] = tmp;
        }
        __syncthreads();
    }

    if (gid < n)
        data[gid] = s[tid];
}

__global__ void bitonicGlobalShared(int* data, int n, int j, int k) {
    __shared__ int s[(BLOCK << 1) + 1];

    int tid = threadIdx.x;
    int base = blockIdx.x << 9;
    int gid = base + tid;

    if (gid < n)
        s[tid] = data[gid];
    else
        s[tid] = INF;

    __syncthreads();

    if (tid < (BLOCK << 1)) {
        int ixj = tid ^ j;
        if (ixj < (BLOCK << 1) && ixj > tid) {
            int dir = ((gid & k) == 0);
            int a = s[tid];
            int b = s[ixj];
            swapDir(a, b, dir);
            s[tid] = a;
            s[ixj] = b;
        }
    }

    __syncthreads();

    if (gid < n)
        data[gid] = s[tid];
}

__global__ void bitonicGlobal(int* data, int n, int j, int k) {
    int gid = (blockIdx.x << 8) + threadIdx.x;
    int ixj = gid ^ j;

    if (ixj > gid && ixj < n) {
        int dir = ((gid & k) == 0);
        int a = data[gid];
        int b = data[ixj];
        swapDir(a, b, dir);
        data[gid] = a;
        data[ixj] = b;
    }
}

int main() {
    int n;
    fread(&n, sizeof(int), 1, stdin);

    size_t bytes_orig = (size_t)n * sizeof(int);
    int* h_orig = (int*)aligned_alloc(128, bytes_orig);
    fread(h_orig, sizeof(int), n, stdin);

    int n_pow2 = 1;
    while (n_pow2 < n) n_pow2 <<= 1;
    size_t bytes_pow2 = (size_t)n_pow2 * sizeof(int);

    int* h_padded = (int*)aligned_alloc(128, bytes_pow2);
    for (int i = 0; i < n; ++i)
        h_padded[i] = h_orig[i];
    for (int i = n; i < n_pow2; ++i)
        h_padded[i] = INF;

    int* d;
    cudaMalloc(&d, bytes_pow2);
    cudaMemcpy(d, h_padded, bytes_pow2, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int blocks_local = (n_pow2 + LOCAL - 1) >> 10;
    oddEvenShared << <blocks_local, LOCAL, 0, stream >> > (d, n_pow2);
    cudaDeviceSynchronize();

    for (int k = LOCAL << 1; k <= n_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j <= BLOCK) {
                int grid = (n_pow2 + (BLOCK << 1) - 1) >> 9;
                bitonicGlobalShared << <grid, BLOCK << 1, 0, stream >> > (d, n_pow2, j, k);
            }
            else {
                int grid = (n_pow2 + BLOCK - 1) >> 8;
                bitonicGlobal << <grid, BLOCK, 0, stream >> > (d, n_pow2, j, k);
            }
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_orig, d, bytes_orig, cudaMemcpyDeviceToHost);
    fwrite(h_orig, sizeof(int), n, stdout);

    cudaStreamDestroy(stream);
    cudaFree(d);
    free(h_orig);
    free(h_padded);
    return 0;
}