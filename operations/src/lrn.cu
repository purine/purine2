// Copyright Lin Min 2015
#include "caffeine/caffeine.hpp"
#include "operations/include/lrn.hpp"

namespace purine {

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype, bool add>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* in,
    const Dtype* scale, const Dtype negative_beta, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    if (add) {
      out[index] += in[index] * pow(scale[index], negative_beta);
    } else {
      out[index] = in[index] * pow(scale[index], negative_beta);
    }
  }
}

LRN::LRN(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(alpha, beta, size) = args;
  CHECK_EQ(outputs_[0]->size(), inputs_[0]->size());
  CHECK_EQ(outputs_[0]->size(), inputs_[1]->size());
}

void LRN::compute_gpu(const vector<bool>& add) {
  Size s = inputs_[0]->size();
  const DTYPE* bottom_data = inputs_[0]->gpu_data();
  const DTYPE* scale_data = inputs_[1]->gpu_data();
  DTYPE* top_data = outputs_[0]->mutable_gpu_data();
  int n_threads = s.count();
  if (add[0] == false) {
    LRNComputeOutput<DTYPE, false><<<CAFFE_GET_BLOCKS(n_threads),
        CAFFE_CUDA_NUM_THREADS, 0, stream()>>>(
            n_threads, bottom_data, scale_data, -beta, top_data);
  } else {
    LRNComputeOutput<DTYPE, true><<<CAFFE_GET_BLOCKS(n_threads),
        CAFFE_CUDA_NUM_THREADS, 0, stream()>>>(
            n_threads, bottom_data, scale_data, -beta, top_data);
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    Dtype* scale) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

LRNScale::LRNScale(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(alpha, beta, size) = args;
  CHECK_EQ(outputs_[0]->size(), inputs_[0]->size());
}

void LRNScale::compute_gpu(const vector<bool>& add) {
  Size s = inputs_[0]->size();
  const DTYPE* bottom_data = inputs_[0]->gpu_data();
  DTYPE* top_data = outputs_[0]->mutable_gpu_data();
  int n_threads = s.num() * s.height() * s.width();
  LRNFillScale<DTYPE><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS, 0,
      stream()>>>(
      n_threads, bottom_data, s.num(), s.channels(), s.height(), s.width(),
      size, alpha / size, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, bool add>
__global__ void LRNComputeDiff(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_data, const Dtype* scale, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio,
    Dtype* bottom_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      if (add) {
        bottom_diff[(head - post_pad) * step] +=
            top_diff[(head - post_pad) * step] *
            pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
            bottom_data[(head - post_pad) * step] * accum_ratio;
      } else {
        bottom_diff[(head - post_pad) * step] =
            top_diff[(head - post_pad) * step] *
            pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
            bottom_data[(head - post_pad) * step] * accum_ratio;
      }
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      if (add) {
        bottom_diff[(head - post_pad) * step] +=
            top_diff[(head - post_pad) * step] *
            pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
            bottom_data[(head - post_pad) * step] * accum_ratio;
      } else {
        bottom_diff[(head - post_pad) * step] =
            top_diff[(head - post_pad) * step] *
            pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
            bottom_data[(head - post_pad) * step] * accum_ratio;
      }
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      if (add) {
        bottom_diff[(head - post_pad) * step] +=
            top_diff[(head - post_pad) * step] *
            pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
            bottom_data[(head - post_pad) * step] * accum_ratio;
      } else {
        bottom_diff[(head - post_pad) * step] =
            top_diff[(head - post_pad) * step] *
            pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
            bottom_data[(head - post_pad) * step] * accum_ratio;
      }
      ++head;
    }
  }
}

LRNDown::LRNDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(alpha, beta, size) = args;
  CHECK_EQ(outputs_[0]->size(), inputs_[0]->size());
  CHECK_EQ(outputs_[0]->size(), inputs_[1]->size());
  CHECK_EQ(outputs_[0]->size(), inputs_[2]->size());
  CHECK_EQ(outputs_[0]->size(), inputs_[3]->size());
}

void LRNDown::compute_gpu(const vector<bool>& add) {
  Size s = inputs_[0]->size();
  int n_threads = s.num() * s.height() * s.width();
  if (add[0] == false) {
    LRNComputeDiff<DTYPE, false><<<CAFFE_GET_BLOCKS(n_threads),
        CAFFE_CUDA_NUM_THREADS, 0, stream()>>>(
            n_threads, inputs_[0]->gpu_data(), inputs_[3]->gpu_data(),
            inputs_[2]->gpu_data(), inputs_[1]->gpu_data(), s.num(),
            s.channels(), s.height(), s.width(), size, -beta,
            DTYPE(2. * alpha * beta / size), outputs_[0]->mutable_gpu_data());
  } else {
    LRNComputeDiff<DTYPE, true><<<CAFFE_GET_BLOCKS(n_threads),
        CAFFE_CUDA_NUM_THREADS, 0, stream()>>>(
            n_threads, inputs_[0]->gpu_data(), inputs_[3]->gpu_data(),
            inputs_[2]->gpu_data(), inputs_[1]->gpu_data(), s.num(),
            s.channels(), s.height(), s.width(), size, -beta,
            DTYPE(2. * alpha * beta / size), outputs_[0]->mutable_gpu_data());
  }
}

}
