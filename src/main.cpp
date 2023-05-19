#include <iostream>
#include <math.h>
#include <vector>

/**
 * The base node for the computation graph.
 * This represents an operation (e.g. addition, multiplication, etc.)
*/
struct Op {
    float data, grad;
    std::vector<Op*> inputs;

    Op(float data, float grad = 0)
        : data(data), grad(grad) {}
    
    void AddInput(Op* input) { inputs.push_back(input); }

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
    virtual float GradFn(Op* input) = 0;
};

struct Add: public Op {
    Op *a, *b;

    Add(Op* a, Op* b)
        : Op(0), a(a), b(b) {}

    float Forward() override {
        data = a->Forward() + b->Forward();
        return data;
    }

private:
    float GradFn(Op* input) override {
        return 1.0f;
    }
};

struct Mul: public Op {
    Mul(Op* a, Op* b)
        : Op(0) {
        AddInput(a);
        AddInput(b);
    }

    float Forward() override {
        data = inputs[0]->Forward() * inputs[1]->Forward();
        return data;
    }

private:
    float GradFn(Op* input) override {
        if (input == inputs[0]) {
            return inputs[1]->data;
        }
        return inputs[0]->data;
    }
};


struct Variable: public Op {
    Variable(float data)
        : Op(data) {}

    float Forward() override {
        return data;
    }

    float GradFn(Op* input) override {
        return 1.0f;
    }
};

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
            Forward();
        }
        return data * (1 - data);
    }
};


int main() {
    Variable a(1), b(234);
    Mul c(&a, &b);
    Sigmoid d(&c);
    d.Backward(1);

    std::cout << a.grad << " " << d.grad << std::endl;
}
