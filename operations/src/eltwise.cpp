// Copyright Lin Min 2015
#include "operations/include/eltwise.hpp"
#include "caffeine/math_functions.hpp"

namespace purine {

Mul::Mul(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args)
    : Operation(inputs, outputs) {
  CHECK_GE(inputs_.size(), 2);
  for (Tensor* input : inputs_) {
    CHECK_EQ(input->size(), outputs_[0]->size());
  }
}

void Mul::compute_cpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  caffe::caffe_mul<DTYPE>(inputs_[0]->size().count(), inputs_[0]->cpu_data(),
      inputs_[1]->cpu_data(), outputs_[0]->mutable_cpu_data());
  for (int i = 2; i < inputs_.size(); ++i) {
    caffe::caffe_mul<DTYPE>(outputs_[0]->size().count(),
        outputs_[0]->cpu_data(), inputs_[i]->cpu_data(),
        outputs_[0]->mutable_cpu_data());
  }
}

void Mul::compute_gpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  caffe::caffe_gpu_mul<DTYPE>(inputs_[0]->size().count(),
      inputs_[0]->gpu_data(), inputs_[1]->gpu_data(),
      outputs_[0]->mutable_gpu_data());
  for (int i = 2; i < inputs_.size(); ++i) {
    caffe::caffe_gpu_mul<DTYPE>(outputs_[0]->size().count(),
        outputs_[0]->gpu_data(), inputs_[i]->gpu_data(),
        outputs_[0]->mutable_gpu_data());
  }
}

Sum::Sum(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
         const param_tuple& args)
    : Operation(inputs, outputs) {
  CHECK_GE(inputs_.size(), 2);
  for (Tensor* input : inputs_) {
    CHECK_EQ(input->size(), outputs_[0]->size());
  }
}

void Sum::compute_cpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  int count = inputs_[0]->size().count();
  caffe::caffe_add<DTYPE>(count, inputs_[0]->cpu_data(), inputs_[1]->cpu_data(),
      outputs_[0]->mutable_cpu_data());
  for (int i = 2; i < inputs_.size(); ++i) {
    caffe::caffe_add<DTYPE>(count, inputs_[i]->cpu_data(),
        outputs_[0]->cpu_data(), outputs_[0]->mutable_cpu_data());
  }
}

void Sum::compute_gpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  int count = inputs_[0]->size().count();
  caffe::caffe_gpu_add<DTYPE>(count, inputs_[0]->gpu_data(),
      inputs_[1]->gpu_data(), outputs_[0]->mutable_gpu_data());
  for (int i = 2; i < inputs_.size(); ++i) {
    caffe::caffe_gpu_add<DTYPE>(count, inputs_[i]->gpu_data(),
        outputs_[0]->gpu_data(), outputs_[0]->mutable_gpu_data());
  }
}

WeightedSum::WeightedSum(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(weights_) = args;
  CHECK_GE(inputs_.size(), 2);
  for (Tensor* input : inputs_) {
    CHECK_EQ(input->size(), outputs_[0]->size());
  }
  CHECK_EQ(inputs_.size(), weights_.size());
}

void WeightedSum::compute_cpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  int count = inputs_[0]->size().count();
  caffe::caffe_cpu_scale(count, weights_[0], inputs_[0]->cpu_data(),
      outputs_[0]->mutable_cpu_data());
  for (int i = 1; i < weights_.size(); ++i) {
    caffe::caffe_axpy(count, weights_[i], inputs_[i]->cpu_data(),
        outputs_[0]->mutable_cpu_data());
  }
}

void WeightedSum::compute_gpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  int count = inputs_[0]->size().count();
  caffe::caffe_gpu_scale(count, weights_[0], inputs_[0]->gpu_data(),
      outputs_[0]->mutable_gpu_data());
  for (int i = 1; i < weights_.size(); ++i) {
    caffe::caffe_gpu_axpy(count, weights_[i], inputs_[i]->gpu_data(),
        outputs_[0]->mutable_gpu_data());
  }
}

Average::Average(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  CHECK_GE(inputs_.size(), 2);
  for (Tensor* input : inputs_) {
    CHECK_EQ(input->size(), outputs_[0]->size());
  }
}

void Average::compute_cpu(const vector<bool>& add) {
  int count = inputs_[0]->size().count();
  caffe::caffe_add<DTYPE>(count, inputs_[0]->cpu_data(), inputs_[1]->cpu_data(),
      outputs_[0]->mutable_cpu_data());
  for (int i = 2; i < inputs_.size(); ++i) {
    caffe::caffe_add<DTYPE>(count, inputs_[i]->cpu_data(),
        outputs_[0]->cpu_data(), outputs_[0]->mutable_cpu_data());
  }
  caffe::caffe_scal<DTYPE>(count, DTYPE(1)/DTYPE(inputs_.size()),
      outputs_[0]->mutable_cpu_data());
}

void Average::compute_gpu(const vector<bool>& add) {
  int count = inputs_[0]->size().count();
  caffe::caffe_gpu_add<DTYPE>(count, inputs_[0]->gpu_data(),
      inputs_[1]->gpu_data(), outputs_[0]->mutable_gpu_data());
  for (int i = 2; i < inputs_.size(); ++i) {
    caffe::caffe_gpu_add<DTYPE>(count, inputs_[i]->gpu_data(),
        outputs_[0]->gpu_data(), outputs_[0]->mutable_gpu_data());
  }
  caffe::caffe_gpu_scal<DTYPE>(count, DTYPE(1)/DTYPE(inputs_.size()),
      outputs_[0]->mutable_gpu_data());
}

Scale::Scale(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(scale) = args;
  CHECK_EQ(inputs_.size(), 1);
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
}

void Scale::compute_cpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  caffe::caffe_cpu_scale<DTYPE>(inputs_[0]->size().count(), scale,
      inputs_[0]->cpu_data(), outputs_[0]->mutable_cpu_data());
}

void Scale::compute_gpu(const vector<bool>& add) {
  CHECK_EQ(add[0], false);
  caffe::caffe_gpu_scale<DTYPE>(inputs_[0]->size().count(), scale,
      inputs_[0]->gpu_data(), outputs_[0]->mutable_gpu_data());
}

}
