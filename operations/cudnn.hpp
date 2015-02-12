#ifndef PURINE_CUDNN
#define PURINE_CUDNN

#include <glog/logging.h>
#include <cudnn.h>

#include "common/common.hpp"
#include "common/cuda.hpp"
#include "operations/size.hpp"

using namespace purine;

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc, Size size,
    Stride stride) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
          size.num(), size.channels(), size.height(), size.width(),
          stride.nstride(), stride.cstride(), stride.hstride(),
          stride.wstride()));
}

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc, Size size) {
  createTensor4dDesc<Dtype>(desc, size, Stride(size));
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc, Size size) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
          size.num(), size.channels(), size.height(), size.width()));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv, pad_h, pad_w, stride_h,
          stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool,
    cudnnPoolingMode_t mode, int h, int w, int pad_h, int pad_w,
    int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool, mode, h, w, pad_h, pad_w,
          stride_h, stride_w));
}

}  // namespace cudnn

#endif
