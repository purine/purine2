// Copyright Lin Min 2015
#include "operations/include/mem_copy.hpp"
#include "caffeine/math_functions.hpp"

namespace purine {

MemCopy::MemCopy(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
  CHECK_EQ(inputs_[0]->rank(), outputs_[0]->rank());
}

void MemCopy::compute_cpu(const vector<bool>& add) {
  if (inputs_[0]->cpu_data() == outputs_[0]->cpu_data()) {
    return;
  } else {
    caffe::caffe_cpu_copy<DTYPE>(inputs_[0]->size().count(),
        inputs_[0]->cpu_data(), outputs_[0]->mutable_cpu_data());
  }
}

void MemCopy::compute_gpu(const vector<bool>& add) {
  if (inputs_[0]->device() >= 0 && outputs_[0]->device() >=0 &&
      inputs_[0]->device() == outputs_[0]->device() &&
      inputs_[0]->gpu_data() == outputs_[0]->mutable_gpu_data()) {
    return;
  }
  const DTYPE* src = inputs_[0]->device() < 0 ? inputs_[0]->cpu_data()
      : inputs_[0]->gpu_data();;
  DTYPE* dst = outputs_[0]->device() < 0 ? outputs_[0]->mutable_cpu_data()
      : outputs_[0]->mutable_gpu_data();
  CUDA_CHECK(cudaMemcpyAsync(dst, src, inputs_[0]->size().count()
          * sizeof(DTYPE), cudaMemcpyDefault, stream()));
}

}
