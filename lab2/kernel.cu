#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <iostream>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void kernel(cudaTextureObject_t tex, uchar4 * out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {            
            float u = ((float)x + 0.5f) / w;
            float v = ((float)y + 0.5f) / h;
            float u_next = ((float)(x + 1) + 0.5f) / w;
            float v_next = ((float)(y + 1) + 0.5f) / h;

            uchar4 p00 = tex2D<uchar4>(tex, u, v);
            uchar4 p01 = tex2D<uchar4>(tex, u, v_next);
            uchar4 p10 = tex2D<uchar4>(tex, u_next, v);
            uchar4 p11 = tex2D<uchar4>(tex, u_next, v_next);

            float y00 = 0.299f * p00.x + 0.587f * p00.y + 0.114f * p00.z;
            float y01 = 0.299f * p01.x + 0.587f * p01.y + 0.114f * p01.z;
            float y10 = 0.299f * p10.x + 0.587f * p10.y + 0.114f * p10.z;
            float y11 = 0.299f * p11.x + 0.587f * p11.y + 0.114f * p11.z;

            float gx = y11 - y00;
            float gy = y10 - y01;
            float magnitude = sqrtf(gx * gx + gy * gy);

            unsigned char edge = (unsigned char)(fminf(magnitude, 255.0f));
            out[y * w + x] = make_uchar4(edge, edge, edge, p00.w);            
        }
    }
}

int main() {
    int w, h;
    std::string input_path, output_path;
    std::cin >> input_path >> output_path;
    FILE* fp = fopen(input_path.c_str(), "rb");
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
    texDesc.addressMode[0] = cudaAddressModeMirror;
    texDesc.addressMode[1] = cudaAddressModeMirror;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4* dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    dim3 block(8, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    kernel << <grid, block >> > (tex, dev_out, w, h);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(output_path.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}