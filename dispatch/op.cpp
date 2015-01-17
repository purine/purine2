// Copyright Lin Min 2015
#include <deque>
#include "dispatch/looper.hpp"
#include "dispatch/op.hpp"

using std::deque;

namespace purine {

Op_::Op_(int rank, int device, const string& thread)
    : Node(rank, device), thread_(thread) {
}

Op_::~Op_() {
}

// find the loop from looper.
// cache in the loop_ variable, next time would be faster.
Loop& Op_::loop() {
  CHECK_EQ(rank_, current_rank());
  if (loop_ == NULL) {
    loop_ = &Looper::task_loop(device_, thread_);
  }
  return *loop_;
}

void Op_::run() {
  if (!o_) {
    setup();
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

}
