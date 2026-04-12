#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#define ll long long
#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)



__global__ void mahalanobis_kernel(uchar4* d_pixels, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    uchar4 pixel = d_pixels[idx];

    int best_class = 0;
    double best_dist = DBL_MAX;
    double p[3];
    p[0] = pixel.x;
    p[1] = pixel.y;
    p[2] = pixel.z;
    for (int c = 0; c < dev_num_classes; ++c) {
        double diff[3] = {
            p[0] - dev_means[c][0],
            p[1] - dev_means[c][1],
            p[2] - dev_means[c][2]
        };

        double temp[3];
        for (int i = 0; i < 3; ++i) {
            temp[i] = 0.0;
            for (int j = 0; j < 3; ++j)
                temp[i] += dev_inv_covs[c][i][j] * diff[j];
        }

        double dist = diff[0] * temp[0] + diff[1] * temp[1] + diff[2] * temp[2];

        if (dist < best_dist - 1e-12) {
            best_dist = dist;
            best_class = c;
        }
    }

    pixel.w = best_class;
    d_pixels[idx] = pixel;
}



struct Mat3 {
    double m[3][3];
    Mat3() { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) m[i][j] = 0.0; }
    double* operator[](int i) { return m[i]; }
    const double* operator[](int i) const { return m[i]; }
};

void compute_mean_cov(const std::vector<uchar4>& samples, double mean[3], Mat3& cov) {
    size_t n = samples.size();
    if (n == 0) return;

    mean[0] = mean[1] = mean[2] = 0.0;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) cov[i][j] = 0.0;

    for (const auto& p : samples) {
        mean[0] += p.x;
        mean[1] += p.y;
        mean[2] += p.z;
    }
    double inv_n = 1.0 / n;
    mean[0] *= inv_n;
    mean[1] *= inv_n;
    mean[2] *= inv_n;

    for (const auto& p : samples) {
        double dr = p.x - mean[0];
        double dg = p.y - mean[1];
        double db = p.z - mean[2];
        cov[0][0] += dr * dr;
        cov[0][1] += dr * dg;
        cov[0][2] += dr * db;
        cov[1][0] += dg * dr;
        cov[1][1] += dg * dg;
        cov[1][2] += dg * db;
        cov[2][0] += db * dr;
        cov[2][1] += db * dg;
        cov[2][2] += db * db;
    }

    if (n > 1) {
        double inv_n1 = 1.0 / (n - 1);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cov[i][j] *= inv_n1;
    }
}

bool invert3x3(const Mat3& m, Mat3& inv) {
    double det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (std::fabs(det) < 1e-12) return false;
    double invdet = 1.0 / det;
    inv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
    inv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invdet;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invdet;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invdet;
    return true;
}

class ClassInfo {
public:
    int mark;
    double mean[3];
    Mat3 cov;
    Mat3 cov_inv;          
};

void load_class(std::vector<ClassInfo>& classes, std::vector<uchar4>& data, int i, int w, int h) {
    long long train_count;
    std::cin >> train_count;

    std::vector<uchar4> v = std::vector<uchar4>(train_count);

    for (int i = 0; i < train_count; i++) {
        int x, y;
        std::cin >> x >> y;
        v[i] = data[w * y + x];
    }

    auto c = ClassInfo();
    c.mark = i;
    compute_mean_cov(v, c.mean, c.cov);
    invert3x3(c.cov, c.cov_inv);
    classes[i] = c;
}

__constant__ double dev_means[32][3];
__constant__ double dev_inv_covs[32][3][3];
__constant__ int dev_num_classes;

int main()
{
    int w, h;
    std::string input_path, output_path;
    std::cin >> input_path >> output_path;

    std::ifstream fp(input_path, std::ios::binary);
    if (!fp) {
        return 1;
    }

    fp.read(reinterpret_cast<char*>(&w), sizeof(int));
    fp.read(reinterpret_cast<char*>(&h), sizeof(int));
    if (!fp) {
        return 1;
    }

    std::vector<uchar4> data(w * h);
    fp.read(reinterpret_cast<char*>(data.data()), sizeof(uchar4) * w * h);
    if (!fp) {
        return 1;
    }

    int class_count;
    std::cin >> class_count;

    std::vector<ClassInfo> classes(class_count);

    for (int i = 0; i < class_count; i++) {
        load_class(classes, data, i, w, h);
    }

    double means[32][3];
    double inv_covs[32][3][3];
    int num_classes = class_count;

    for (int i = 0; i < class_count; i++) {
        means[i][0] = classes[i].mean[0];
        means[i][1] = classes[i].mean[1];
        means[i][2] = classes[i].mean[2];

        for (int x = 0; x < 3; x++) {
            for (int y = 0; x < 3; x++) {
                inv_covs[i][x][y] = classes[i].cov_inv.m[x][y];
        }        
    }

    cudaMemcpyToSymbol(dev_means, means, sizeof(double) * 32 * 3);
    cudaMemcpyToSymbol(dev_inv_covs, inv_covs, sizeof(double) * 32 * 3 * 3);
    cudaMemcpyToSymbol(&dev_num_classes, &num_classes, sizeof(int));

    int pixel_bytes = sizeof(uchar4) * data.size();
    uchar4* dev_pixels = nullptr;

    cudaMalloc(&dev_pixels, pixel_bytes);

    cudaMemcpy(dev_pixels, data.data(), pixel_bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (w * h + threads_per_block - 1) / threads_per_block;

    mahalanobis_kernel << <blocks, threads_per_block >> > (dev_pixels, w, h);
    cudaDeviceSynchronize();

    std::vector<uint32_t> h_pixels(w * h);
    cudaMemcpy(h_pixels.data(), dev_pixels, pixel_bytes, cudaMemcpyDeviceToHost);


    for (size_t i = 0; i < w * h; ++i) {
        uint32_t val = h_pixels[i];
        data[i].x = (val >> 24) & 0xFF;
        data[i].y = (val >> 16) & 0xFF;
        data[i].z = (val >> 8) & 0xFF;
        data[i].w = val & 0xFF;
    }

    cudaFree(dev_pixels);

    fp_out = fopen(output_path.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp_out);
    fwrite(&h, sizeof(int), 1, fp_out);
    fwrite(data.data(), sizeof(uchar4), w * h, fp_out);
    fclose(fp_out);

    return 0;
}

