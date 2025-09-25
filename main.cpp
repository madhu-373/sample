#include "math_header.h"

#include<iostream>
using namespace std;

int main()
{
    cout<<"Hello, World!"<< endl;

    int sum = add(5, 3);
    int product = multiply(4, 7);
    double area = calculateArea(2.5);
    
    printResult(sum);
    printResult(product);
    std::cout << "Area: " << area << std::endl;
    
    return 0;

} 