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
void Op<O>::run() {
  if (!fn_) {
    vector<Tensor*> input_tensors(inputs_.size());
    vector<Tensor*> output_tensors(outputs_.size());
    transform(inputs_.begin(), inputs_.end(), input_tensors.begin(),
        [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
    transform(outputs_.begin(), outputs_.end(), output_tensors.begin(),
        [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
    o_.reset(new O(input_tensors, output_tensors, args_));
  }
  loop().post([this](){
        vector<bool> add(outputs_.size());
        transform(outputs_.begin(), outputs_.end(), add.begin(),
            [] (Node* b) -> bool { return b->in() > 0; });
        if (device_ < 0) {
          for (Node* node : inputs_) {
            Blob* b = static_cast<Blob*>(node);
            if (b->cuda_event() == NULL) {
              continue;
            }
            CUDA_CHECK(cudaEventSynchronize(b->cuda_event()));
          }
          o_->compute_cpu(add);
        } else {
          for (Node* node : inputs_) {
            Blob* b = static_cast<Blob*>(node);
            if (b->cuda_event() == NULL) {
              continue;
            }
            CUDA_CHECK(cudaStreamWaitEvent(stream(), b->cuda_event(), 0));
          }
          o_->compute_gpu(add);
        }
        for (Node* output : outputs_) {
          output->inc_in();
        }
      });
  // ++sink_counter if is sink
  if (outputs_.size() == 0) {
    loop().post([this](){
          if (device_ >= 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream()));
          }
          ++(cached_root_->sink_counter());
        });
  }
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
