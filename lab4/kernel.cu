#include <thrust/device_vector.h>
#include <thrust/extrema.h>

int main() {
    thrust::device_vector<int> v(10);
    auto it = thrust::max_element(v.begin(), v.end());
}