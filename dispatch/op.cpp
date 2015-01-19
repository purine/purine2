// Copyright Lin Min 2015
#include <deque>
#include "dispatch/looper.hpp"
#include "dispatch/op.hpp"
#include "dispatch/blob.hpp"

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

Op<Irecv>::Op(const typename Irecv::param_tuple& args,
    int rank, int device, const string& thread)
    : Op_(rank, device, thread), args_(args) {
  mpi_test_ = [this] () {
    MPI_Request* request = static_cast<Irecv*>(o_.get())->mpi_request();
    int flag;
    MPI_CHECK(MPI_Test(request, &flag, MPI_STATUS_IGNORE));
    if (flag) {
      for (Node* output : outputs_) {
        output->inc_in();
      }
      // ++sink_counter if is sink
      if (outputs_.size() == 0) {
        ++(cached_root_->sink_counter());
      }
    } else {
      loop().post(mpi_test_);
    }
  };
}

void Op<Irecv>::setup() {
  vector<Tensor*> input_tensors(this->inputs_.size());
  vector<Tensor*> output_tensors(this->outputs_.size());
  transform(this->inputs_.begin(), this->inputs_.end(), input_tensors.begin(),
      [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
  transform(this->outputs_.begin(), this->outputs_.end(),
      output_tensors.begin(), [] (Node* b) -> Tensor*
      { return static_cast<Blob*>(b)->tensor(); });
  this->o_.reset(new Irecv(input_tensors, output_tensors, this->args_));
}

void Op<Irecv>::run() {
  if (!o_) {
    setup();
  }
  loop().post([this]() {
        vector<bool> add(outputs_.size());
        transform(outputs_.begin(), outputs_.end(), add.begin(),
            [] (Node* b) -> bool { return b->in() > 0; });
        if (device_ < 0) {
          o_->compute_cpu(add);
          loop().post(mpi_test_);
        } else {
          LOG(FATAL) << "current version of Purine does not support this";
        }
      });
}


Op<Isend>::Op(const typename Isend::param_tuple& args,
    int rank, int device, const string& thread)
    : Op_(rank, device, thread), args_(args) {
  mpi_test_ = [this]() {
    MPI_Request* request = static_cast<Isend*>(o_.get())->mpi_request();
    int flag;
    MPI_CHECK(MPI_Test(request, &flag, MPI_STATUS_IGNORE));
    if (flag) {
      ++(cached_root_->sink_counter());
    } else {
      loop().post(mpi_test_);
    }
  };
}

void Op<Isend>::setup() {
  vector<Tensor*> input_tensors(this->inputs_.size());
  vector<Tensor*> output_tensors(this->outputs_.size());
  transform(this->inputs_.begin(), this->inputs_.end(), input_tensors.begin(),
      [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
  transform(this->outputs_.begin(), this->outputs_.end(),
      output_tensors.begin(), [] (Node* b) -> Tensor*
      { return static_cast<Blob*>(b)->tensor(); });
  this->o_.reset(new Isend(input_tensors, output_tensors, this->args_));
}

void Op<Isend>::run() {
  if (!o_) {
    setup();
  }
  loop().post([this]() {
        // there is not output for Isend
        if (device_ < 0) {
          o_->compute_cpu({});
          loop().post(mpi_test_);
        } else {
          LOG(FATAL) << "current version of Purine does not support this";
        }
      });
}

Op_& operator >> (const vector<Blob*>& inputs, Op_& op) {
  for (Blob* input : inputs) {
    input->outputs_.push_back(&op);
    op.inputs_.push_back(input);
  }
  return op;
}

void operator >> (Op_& op, const vector<Blob*>& outputs) {
  for (Blob* output : outputs) {
    output->inputs_.push_back(&op);
    op.outputs_.push_back(output);
  }
  return;
}

}
