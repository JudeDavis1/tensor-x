#include "base_ops.h"


struct Sigmoid: public Op {
    Op* input;
    bool did_forward_pass = false;

    Sigmoid(Op* input)
        : Op(0), input(input) {
        AddInput(input);
    }
    
    float Forward() override {
        data = 1 / (1 + exp(-input->Forward()));
        did_forward_pass = true;
        return data;
    }

    float GradFn(Op* input) override {
        if (!did_forward_pass) {
            this->Forward();
        }
        return data * (1 - data);
    }
};