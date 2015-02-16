// Copyright Lin Min 2015
#include <deque>
#include "dispatch/op.hpp"
#include "dispatch/blob.hpp"
#include "dispatch/runnable.hpp"

using std::deque;

namespace purine {

Op_::Op_(int rank, int device, const string& thread)
    : Node(rank, device), thread_(thread) {
}

Op_::~Op_() {
}

// find the loop from looper.
// cache in the loop_ variable, next time would be faster.
LoopInterface& Op_::loop() {
  CHECK_EQ(rank_, current_rank());
  if (loop_ == NULL) {
    loop_ = &(dynamic_cast<Runnable*>(cached_root_)->
        task_loop(device_, thread_));
  }
  return *loop_;
}

void Op_::compute() {
  loop().post([this](){
        // put setup code inside..
        if (!this->o_) {
          this->setup();
        }
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
#ifndef NDEBUG
          LOG(INFO) << "start " << cached_name_;
#endif
          o_->compute_cpu(add);
        } else {
          for (Node* node : inputs_) {
            Blob* b = static_cast<Blob*>(node);
            if (b->cuda_event() == NULL) {
              continue;
            }
            CUDA_CHECK(cudaStreamWaitEvent(stream(), b->cuda_event(), 0));
          }
#ifndef NDEBUG
          LOG(INFO) << "start " << cached_name_;
#endif
          o_->compute_gpu(add);
#ifndef NDEBUG
          CUDA_CHECK(cudaStreamSynchronize(stream()));
#endif
        }
        for (Node* output : outputs_) {
          output->inc_in();
        }
        // ++sink_counter if is sink
        if (outputs_.size() == 0) {
          loop().post([this]()->void{
                if (device_ >= 0) {
                  CUDA_CHECK(cudaStreamSynchronize(stream()));
                }
                ++(dynamic_cast<Runnable*>(cached_root_)->sink_counter());
              });
        }
      });
}

void Op_::set_inputs(const vector<Blob*>& inputs) {
  CHECK(!this->input_setup_);
  for (Blob* input : inputs) {
    input->outputs_.push_back(this);
    this->inputs_.push_back(input);
  }
  this->input_setup_ = true;
}

void Op_::set_outputs(const vector<Blob*>& outputs) {
  CHECK(!this->output_setup_);
  for (Blob* output : outputs) {
    output->inputs_.push_back(this);
    this->outputs_.push_back(output);
  }
  this->output_setup_ = true;
}

void Op_::check_inputs(const vector<Blob*>& inputs) {
  for (Blob* input : inputs) {
    CHECK_EQ(input->rank(), rank_);
    CHECK_EQ(input->device(), device_);
  }
}

void Op_::check_outputs(const vector<Blob*>& outputs) {
  for (Blob* output : outputs) {
    CHECK_EQ(output->rank(), rank_);
    CHECK_EQ(output->device(), device_);
  }
}

Op_& operator >> (const vector<Blob*>& inputs, Op_& op) {
  op.check_inputs(inputs);
  op.set_inputs(inputs);
  return op;
}

const vector<Blob*>& operator >> (Op_& op, const vector<Blob*>& outputs) {
  op.check_outputs(outputs);
  op.set_outputs(outputs);
  return outputs;
}

Op<Irecv>::Op(int rank, int device, const string& thread,
    const typename Irecv::param_tuple& args)
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
        ++(dynamic_cast<Runnable*>(cached_root_)->sink_counter());
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

void Op<Irecv>::compute() {
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


Op<Isend>::Op(int rank, int device, const string& thread,
    const typename Isend::param_tuple& args)
    : Op_(rank, device, thread), args_(args) {
  mpi_test_ = [this]() {
    MPI_Request* request = static_cast<Isend*>(o_.get())->mpi_request();
    int flag;
    MPI_CHECK(MPI_Test(request, &flag, MPI_STATUS_IGNORE));
    if (flag) {
      ++(dynamic_cast<Runnable*>(cached_root_)->sink_counter());
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

void Op<Isend>::compute() {
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

Op<MemCopy>::Op(int rank, int device, const string& thread,
    const typename MemCopy::param_tuple& args) : Op_(rank, device, thread) {
  // rank_ and device_ is reset when setup.
}

void Op<MemCopy>::setup() {
  CHECK_EQ(inputs_.size(), 1);
  CHECK_EQ(outputs_.size(), 1);
  CHECK_EQ(inputs_[0]->rank(), outputs_[0]->rank());
  this->o_.reset(new MemCopy({static_cast<Blob*>(inputs_[0])->tensor()},
          {static_cast<Blob*>(outputs_[0])->tensor()}, tuple<>()));
}

void Op<MemCopy>::set_inputs(const vector<Blob*>& inputs) {
  Op_::set_inputs(inputs);
  if (input_setup_ && output_setup_) {
    rank_ = inputs_[0]->rank();
    if (inputs_[0]->device() >= 0 || outputs_[0]->device() < 0) {
      device_ = inputs_[0]->device();
    } else {
      device_ = outputs_[0]->device();
    }
  }
}

void Op<MemCopy>::set_outputs(const vector<Blob*>& outputs) {
  Op_::set_outputs(outputs);
  if (output_setup_ && input_setup_) {
    rank_ = inputs_[0]->rank();
    if (inputs_[0]->device() >= 0 || outputs_[0]->device() < 0) {
      device_ = inputs_[0]->device();
    } else {
      device_ = outputs_[0]->device();
    }
  }
}

}
