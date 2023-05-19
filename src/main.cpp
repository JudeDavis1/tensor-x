#include <iostream>
#include <math.h>

#include "activation_ops.h"


int main() {
    Variable a(1), b(234);
    Mul c(&a, &b);
    Sigmoid d(&c);
    d.Backward(1);

    std::cout << a.grad << " " << c.grad << std::endl;
}
