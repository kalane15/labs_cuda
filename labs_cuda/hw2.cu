#include <iostream>
#include <iomanip>

void bubble_sort(int n, float* arr) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int n;
    std::cin >> n;
    float* arr = new float[n];
    for (int i = 0; i < n; i++) {
        std::cin >> arr[i];
    }
    bubble_sort(n, arr);
    for (int i = 0; i < n; i++) {
        std::cout << std::scientific << std::setprecision(6) << arr[i];
        if (i < n - 1) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
    delete[] arr;
    return 0;
}