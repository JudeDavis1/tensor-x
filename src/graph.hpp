#pragma once

#include <vector>
#include <unordered_map>

#include "base_ops.hpp"


using NodeList = std::vector<BaseOp*>;

class Graph {
public:
    NodeList nodes;
    BaseOp* head;
    std::unordered_map<BaseOp*, NodeList> adjacency_map;

    Graph() { }

    void AddNode(BaseOp* node) {
        if (nodes.size() == 0) {
            head = node;
        }
        nodes.push_back(node);
    }

    void AddEdge(BaseOp* from_node, BaseOp* to_node) {
        adjacency_map[from_node].push_back(to_node);
    }

    void Forward() {
        for (auto& node: nodes) {
            node->Forward();
        }
    }

    void Backward() {
        nodes.back()->Backward(1);
    }
};
