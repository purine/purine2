// Copyright Lin Min 2015
#include "operations/include/dummy.hpp"

namespace purine {

Dummy::Dummy(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
}

void Dummy::compute_cpu(const vector<bool>& add) {
}

void Dummy::compute_gpu(const vector<bool>& add) {
}

}
