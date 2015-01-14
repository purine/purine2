// Copyright Lin Min 2015
#include "caffeine/caffeine.hpp"
#include "caffeine/math_functions.hpp"
#include "operations/include/random.hpp"

namespace purine {

Gaussian::Gaussian(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(mean, std) = args;
}

void Gaussian::compute_cpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_rng_gaussian<DTYPE>(count, mean, std,
        output->mutable_cpu_data());
  }
}

void Gaussian::compute_gpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_gpu_rng_gaussian<DTYPE>(count, mean, std,
        output->mutable_gpu_data());
  }
}

Uniform::Uniform(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(min, max) = args;
}

void Uniform::compute_cpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_rng_uniform<DTYPE>(count, min, max,
        output->mutable_cpu_data());
  }
}

void Uniform::compute_gpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_gpu_rng_uniform<DTYPE>(count, min, max,
        output->mutable_gpu_data());
  }
}

Constant::Constant(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(constant) = args;
}

void Constant::compute_cpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_set<DTYPE>(count, constant, output->mutable_cpu_data());
  }
}

void Constant::compute_gpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_gpu_set<DTYPE>(count, constant, output->mutable_gpu_data());
  }
}

Bernoulli::Bernoulli(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(prob) = args;
}

void Bernoulli::compute_cpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_rng_bernoulli<DTYPE>(count, prob,
        reinterpret_cast<int*>(output->mutable_cpu_data()));
  }
}

void Bernoulli::compute_gpu(const vector<bool>& add) {
  for (Tensor* output : outputs_) {
    int count = output->size().count();
    caffe::caffe_gpu_rng_bernoulli<DTYPE>(count, prob,
        output->mutable_gpu_data());
  }
}

}
