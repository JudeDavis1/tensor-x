#include <iostream>
#include <math.h>

#include "graph.hpp"
#include "activation_ops.hpp"


int main() {
    Graph graph;

    Variable a(1);
    Variable b(2);
    Sigmoid s(&a);

    graph.AddNode(&a);
    graph.AddNode(&b);
    graph.AddNode(&s);

    graph.AddEdge(&a, &s);
    graph.AddEdge(&b, &s);
    graph.Forward();

    graph.Backward();

    std::cout << a.grad << std::endl;
}
