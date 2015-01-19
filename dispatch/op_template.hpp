#ifndef PURINE_OP_TEMPLATE
#define PURINE_OP_TEMPLATE

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "dispatch/op.hpp"
#include "dispatch/blob.hpp"

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
void Op<O>::setup() {
  vector<Tensor*> input_tensors(this->inputs_.size());
  vector<Tensor*> output_tensors(this->outputs_.size());
  transform(this->inputs_.begin(), this->inputs_.end(), input_tensors.begin(),
      [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
  transform(this->outputs_.begin(), this->outputs_.end(),
      output_tensors.begin(), [] (Node* b) -> Tensor*
      { return static_cast<Blob*>(b)->tensor(); });
  this->o_.reset(new O(input_tensors, output_tensors, this->args_));
}

}

#endif
