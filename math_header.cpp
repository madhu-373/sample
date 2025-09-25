#include "math_header.h"
#include <cmath>

// Function definitions
int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

double calculateArea(double radius) {
    return 3.14159 * radius * radius;
}

void printResult(int result) {
    std::cout << "Result: " << result << std::endl;
}