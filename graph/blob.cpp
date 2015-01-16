
#include <deque>
#include "graph/op.hpp"
#include "graph/blob.hpp"

using std::deque;

namespace purine {

Blob::Blob(const Size& s, int rank, int device) : Node(rank, device) {
  tensor_.reset(new Tensor(s, rank, device));
}

Blob::~Blob() {
}

Tensor* Blob::tensor() {
  return tensor_.get();
}

// this is always called from in_thread
void Blob::run() {
  if (inputs_.size() == 0) {
    // this blob is the source of a graph.
    for (Node* out : outputs_) {
      out->inc_in();
    }
    return;
  }
  // record cudaevent if outputs are in different thread as in input.
  // and inputs are executed in GPU.
  int device = static_cast<Op_*>(inputs_[0])->device();
  if (device >= 0) {
    Loop& in_loop = static_cast<Op_*>(inputs_[0])->loop();
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
  // if this blob is a sink
  if (outputs_.size() == 0) {
    if (cuda_event_) {
      CUDA_CHECK(cudaEventSynchronize(cuda_event_));
    }
    // after syncing, update conditional variable.
    ++(cached_root_->sink_counter());
  }
  for (Node* out : outputs_) {
    out->inc_in();
  }
}

}
