#ifndef PURINE_OP_TEMPLATE
#define PURINE_OP_TEMPLATE

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "graph/op.hpp"
#include "graph/blob.hpp"

using std::function;
using std::string;
using std::vector;
using std::shared_ptr;
using std::transform;

namespace purine {

template <typename O>
Op<O>::Op(const typename O::param_tuple& args,
      int rank, int device, const string& thread)
      : Op_(rank, device, thread), args_(args) {
}

template <typename O>
Op<O>::Op(const typename O::param_tuple& args,
      const initializer_list<Blob*>& inputs,
      const initializer_list<Blob*>& outputs,
      int rank, int device, const string& thread)
      : Op(args, rank, device, thread) {
    inputs_ = inputs;
    outputs_ = outputs;
}

template <typename O>
void Op<O>::setup() {
  vector<Tensor*> input_tensors(inputs_.size());
  vector<Tensor*> output_tensors(outputs_.size());
  transform(inputs_.begin(), inputs_.end(), input_tensors.begin(),
      [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
  transform(outputs_.begin(), outputs_.end(), output_tensors.begin(),
      [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
  o_.reset(new O(input_tensors, output_tensors, args_));
}

template <typename O>
Op<O>& operator >> (const vector<Blob*>& inputs, Op<O>& op) {
  for (Blob* input : inputs) {
    input->outputs_.push_back(&op);
    op.inputs_.push_back(input);
  }
  return op;
}

template <typename O>
void operator >> (Op<O>& op, const vector<Blob*>& outputs) {
  for (Blob* output : outputs) {
    output->inputs_.push_back(&op);
    op.outputs_.push_back(output);
  }
  return;
}

}

#endif
