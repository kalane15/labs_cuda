#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void kernel(cudaTextureObject_t tex, uchar4* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    uchar4 p00 = tex2D<uchar4>(tex, (float)x, (float)y);
    uchar4 p01 = tex2D<uchar4>(tex, (float)x, (float)(y + 1));
    uchar4 p10 = tex2D<uchar4>(tex, (float)(x + 1), (float)y);
    uchar4 p11 = tex2D<uchar4>(tex, (float)(x + 1), (float)(y + 1));

    float y00 = 0.299f * p00.x + 0.587f * p00.y + 0.114f * p00.z;
    float y01 = 0.299f * p01.x + 0.587f * p01.y + 0.114f * p01.z;
    float y10 = 0.299f * p10.x + 0.587f * p10.y + 0.114f * p10.z;
    float y11 = 0.299f * p11.x + 0.587f * p11.y + 0.114f * p11.z;

    float gx = y11 - y00;
    float gy = y10 - y01;
    float edge = sqrtf(gx * gx + gy * gy);

    out[y * w + x] = make_uchar4(edge, edge, edge, 0.0);
}

int main() {
    int w, h;
    FILE* fp = fopen("in.data", "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray* arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeMirror;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4* dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>> (tex, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen("out.data", "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}