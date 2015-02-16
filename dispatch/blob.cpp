
#include <deque>
#include "dispatch/op.hpp"
#include "dispatch/blob.hpp"
#include "dispatch/runnable.hpp"

using std::deque;

namespace purine {

Blob::Blob(int rank, int device, const Size& s) : Node(rank, device) {
  tensor_.reset(new Tensor(rank, device, s));
}

Blob::Blob(shared_ptr<Tensor> tensor) : Node(tensor->rank(), tensor->device()) {
  tensor_ = tensor;
}

Blob::~Blob() {
  if (cuda_event_) {
    CUDA_CHECK(cudaEventDestroy(cuda_event_));
  }
}

// this is always called from in_thread
void Blob::compute() {
  if (!inputs_.empty()) {
    // record cudaevent if outputs are in different thread as in input.
    // and inputs are executed in GPU.
    int device = static_cast<Op_*>(inputs_[0])->device();
    if (device >= 0) {
      LoopInterface& in_loop = static_cast<Op_*>(inputs_[0])->loop();
      if (any_of(outputs_.begin(), outputs_.end(),
              [&](Node* output)->bool {
                Op_* op = static_cast<Op_*>(output);
                return &op->loop() != &in_loop;
              }) || outputs_.size() == 0) {
        if (cuda_event_ == NULL) {
          CUDA_CHECK(cudaEventCreate(&cuda_event_,
                  cudaEventBlockingSync|cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(cuda_event_, stream()));
      }
    }
  }
  // if this blob is a sink
  if (outputs_.size() == 0) {
    if (cuda_event_) {
      CUDA_CHECK(cudaEventSynchronize(cuda_event_));
    }
    // after syncing, update conditional variable.
    ++(dynamic_cast<Runnable*>(cached_root_)->sink_counter());
  }
  for (Node* out : outputs_) {
    out->inc_in();
  }
}

void Blob::share_from(Blob* other) {
  CHECK_EQ(other->rank_, rank_);
  CHECK_EQ(other->device_, device_);
  tensor_ = other->tensor_;
}

}
