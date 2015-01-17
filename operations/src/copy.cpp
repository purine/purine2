// Copyright Lin Min 2015
#include "operations/include/copy.hpp"
#include "caffeine/math_functions.hpp"

namespace purine {

Copy::Copy(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
  CHECK_EQ(inputs_[0]->rank(), outputs_[0]->rank());
}

void Copy::compute_cpu(const vector<bool>& add) {
  if (inputs_[0]->cpu_data() == outputs_[0]->cpu_data()) {
    return;
  } else {
    caffe::caffe_cpu_copy<DTYPE>(inputs_[0]->size().count(),
        inputs_[0]->cpu_data(), outputs_[0]->mutable_cpu_data());
  }
}

void Copy::compute_gpu(const vector<bool>& add) {
  const DTYPE* src = inputs_[0]->device() < 0 ? inputs_[0]->cpu_data()
      : inputs_[0]->gpu_data();;
  DTYPE* dst = outputs_[0]->device() < 0 ? outputs_[0]->mutable_cpu_data()
      : outputs_[0]->mutable_gpu_data();
  CUDA_CHECK(cudaMemcpyAsync(dst, src, inputs_[0]->size().count()
          * sizeof(DTYPE), cudaMemcpyDefault, stream()));
}

}
