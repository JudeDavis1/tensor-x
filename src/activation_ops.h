#include "base_ops.h"


struct Sigmoid: public BaseOp {
    BaseOp* input;
    bool did_forward_pass = false;

    Sigmoid(BaseOp* input)
        : BaseOp(0), input(input) {
        AddInput(input);
    }
    
    float Forward() override {
        data = 1 / (1 + exp(-input->Forward()));
        did_forward_pass = true;
        return data;
    }

    float GradFn(BaseOp* input) override {
        if (!did_forward_pass) {
            this->Forward();
        }
        return data * (1 - data);
    }
};
