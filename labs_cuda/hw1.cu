#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

int main()
{
    float a, b, c;
    std::cin >> a >> b >> c;

    std::cout.precision(6);
    std::cout << std::fixed;

    if (a == 0) {
        if (b == 0) {
            if (c == 0) {
                std::cout << "any";
                return;
            }
            std::cout << "incorrect";
            return;
        }
        std::cout << -c / b;
        return;
    }

    float d = b * b - 4 * a * c;
    if (d < 0.0) {
        std::cout << "imaginary";
        return;
    }

    float x1, x2;
    x1 = (-b + sqrt(d)) / (2 * a);
    x2 = (-b - sqrt(d)) / (2 * a);

    std::cout << x1 << " " << x2;
}
