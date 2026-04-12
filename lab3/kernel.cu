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


__global__ void kernel(double* c, const double* a, const double* b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (index < n) {
        c[index] = a[index] - b[index];
        index += offset;
    }
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

    return 0;
}

