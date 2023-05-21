#pragma once

#include <vector>


/**
 * The base node for the computation graph.
 * This represents an operation (e.g. addition, multiplication, etc.)
*/
struct BaseOp {
    float data, grad;
    std::vector<BaseOp*> inputs;

    BaseOp(float data, float grad = 0)
        : data(data), grad(grad) {}
    
    void AddInput(BaseOp* input) { inputs.push_back(input); }

    virtual float Forward() = 0;
    virtual void Backward(float new_grad) {
        grad += new_grad;

        // Propagate the gradient to the inputs
        for (auto& node: inputs) {
            float in_grad = grad * GradFn(node);
            node->Backward(in_grad);
        }
    };

protected:
    virtual float GradFn(BaseOp* input) = 0;
};


/**
 * Addition routine.
*/
struct Add: public BaseOp {
    BaseOp *a, *b;

    Add(BaseOp* a, BaseOp* b)
        : BaseOp(0), a(a), b(b) {}

    float Forward() override {
        data = a->Forward() + b->Forward();
        return data;
    }

private:
    float GradFn(BaseOp* input) override {
        return 1.0f;
    }
};


/**
 * Multiplication routine.
*/
struct Mul: public BaseOp {
    Mul(BaseOp* a, BaseOp* b)
        : BaseOp(0) {
        AddInput(a);
        AddInput(b);
    }

    float Forward() override {
        data = inputs[0]->Forward() * inputs[1]->Forward();
        return data;
    }

private:
    float GradFn(BaseOp* input) override {
        if (input == inputs[0]) {
            return inputs[1]->data;
        }
        return inputs[0]->data;
    }
};


/**
 * A variable node.
 * This is a leaf node in the computation graph.
*/
struct Variable: public BaseOp {
    Variable(float data)
        : BaseOp(data) {}

    float Forward() override {
        return data;
    }

    float GradFn(BaseOp* input) override {
        return 1.0f;
    }
};

