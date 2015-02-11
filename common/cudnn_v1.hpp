#ifndef PURINE_CUDNN_V1
#define PURINE_CUDNN_V1

#include <glog/logging.h>
#include <cudnn.h>
#include <dlfcn.h>
#include "common/cuda.hpp"

namespace purine {

class cudnn_v1 {
 protected:
  void* shared_lib;
 public:
  typedef enum {
    CUDNN_RESULT_ACCUMULATE      = 0,
    CUDNN_RESULT_NO_ACCUMULATE   = 1
  } cudnnAccumulateResult_t;

  typedef cudnnStatus_t CUDNNWINAPI (*cudnnCreate_t)(cudnnHandle_t *handle);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnDestroy_t)(cudnnHandle_t handle);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnSetStream_t)(cudnnHandle_t handle,
      cudaStream_t streamId);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnGetStream_t)(cudnnHandle_t handle,
      cudaStream_t *streamId);

  typedef cudnnStatus_t CUDNNWINAPI (*cudnnCreateTensor4dDescriptor_t)(
      cudnnTensorDescriptor_t *tensorDesc);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnSetTensor4dDescriptor_t)(
      cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
      cudnnDataType_t dataType, int n, int c, int h, int w);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnSetTensor4dDescriptorEx_t)(
      cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n,
      int c, int h, int w, int nStride, int cStride, int hStride, int wStride);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnDestroyTensor4dDescriptor_t)(
      cudnnTensorDescriptor_t tensorDesc);

  typedef cudnnStatus_t CUDNNWINAPI (*cudnnCreateFilterDescriptor_t)(
      cudnnFilterDescriptor_t *filterDesc);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnSetFilterDescriptor_t)(
      cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, int k,
      int c, int h, int w);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnDestroyFilterDescriptor_t)(
      cudnnFilterDescriptor_t filterDesc);

  typedef cudnnStatus_t CUDNNWINAPI (*cudnnCreateConvolutionDescriptor_t)(
      cudnnConvolutionDescriptor_t *convDesc);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnSetConvolutionDescriptor_t)(
      cudnnConvolutionDescriptor_t convDesc,
      cudnnTensorDescriptor_t inputTensorDesc,
      cudnnFilterDescriptor_t filterDesc, int pad_h, int pad_w, int u, int v,
      int upscalex, int upscaley, cudnnConvolutionMode_t mode);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnSetConvolutionDescriptorEx_t)(
      cudnnConvolutionDescriptor_t convDesc, int n, int c, int h, int w, int k,
      int r, int s, int pad_h, int pad_w, int u, int v,
      int upscalex, int upscaley, cudnnConvolutionMode_t mode);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnDestroyConvolutionDescriptor_t)(
      cudnnConvolutionDescriptor_t convDesc);

  typedef cudnnStatus_t CUDNNWINAPI (*cudnnConvolutionForward_t)(cudnnHandle_t handle,
      cudnnTensorDescriptor_t srcDesc, const void *srcData,
      cudnnFilterDescriptor_t filterDesc, const void *filterData,
      cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t destDesc,
      void *destData, cudnnAccumulateResult_t accumulate);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnConvolutionBackwardBias_t)(
      cudnnHandle_t handle, cudnnTensorDescriptor_t srcDesc,
      const void *srcData, cudnnTensorDescriptor_t destDesc, void *destData,
      cudnnAccumulateResult_t accumulate);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnConvolutionBackwardData_t)(
      cudnnHandle_t handle, cudnnFilterDescriptor_t filterDesc,
      const void *filterData, cudnnTensorDescriptor_t diffDesc,
      const void *diffData, cudnnConvolutionDescriptor_t convDesc,
      cudnnTensorDescriptor_t gradDesc, void *gradData,
      cudnnAccumulateResult_t accumulate);
  typedef cudnnStatus_t CUDNNWINAPI (*cudnnConvolutionBackwardFilter_t)(
      cudnnHandle_t handle, cudnnTensorDescriptor_t srcDesc,
      const void *srcData, cudnnTensorDescriptor_t diffDesc,
      const void *diffData, cudnnConvolutionDescriptor_t convDesc,
      cudnnFilterDescriptor_t gradDesc, void *gradData,
      cudnnAccumulateResult_t accumulate);

  cudnn_v1() {
    shared_lib = dlopen("libcudnn_v1.so", RTLD_LAZY);
    CHECK(shared_lib);
    cudnnCreate = (cudnnCreate_t)dlsym(shared_lib, "cudnnCreate");
    cudnnDestroy = (cudnnDestroy_t)dlsym(shared_lib, "cudnnDestroy");
    cudnnSetStream = (cudnnSetStream_t)dlsym(shared_lib, "cudnnSetStream");
    cudnnGetStream = (cudnnGetStream_t)dlsym(shared_lib, "cudnnGetStream");
    cudnnCreateTensor4dDescriptor = (cudnnCreateTensor4dDescriptor_t)dlsym(shared_lib,
        "cudnnCreateTensor4dDescriptor");
    cudnnSetTensor4dDescriptor = (cudnnSetTensor4dDescriptor_t)dlsym(shared_lib,
        "cudnnSetTensor4dDescriptor");
    cudnnSetTensor4dDescriptorEx = (cudnnSetTensor4dDescriptorEx_t)dlsym(shared_lib,
        "cudnnSetTensor4dDescriptorEx");
    cudnnDestroyTensor4dDescriptor = (cudnnDestroyTensor4dDescriptor_t)dlsym(shared_lib,
        "cudnnDestroyTensor4dDescriptor");
    cudnnCreateFilterDescriptor = (cudnnCreateFilterDescriptor_t)dlsym(shared_lib,
        "cudnnCreateFilterDescriptor");
    cudnnSetFilterDescriptor = (cudnnSetFilterDescriptor_t)dlsym(shared_lib,
        "cudnnSetFilterDescriptor");
    cudnnDestroyFilterDescriptor = (cudnnDestroyFilterDescriptor_t)dlsym(shared_lib,
        "cudnnDestroyFilterDescriptor");
    cudnnCreateConvolutionDescriptor = (cudnnCreateConvolutionDescriptor_t)dlsym(shared_lib,
        "cudnnCreateConvolutionDescriptor");
    cudnnSetConvolutionDescriptor = (cudnnSetConvolutionDescriptor_t)dlsym(shared_lib,
        "cudnnSetConvolutionDescriptor");
    cudnnSetConvolutionDescriptorEx = (cudnnSetConvolutionDescriptorEx_t)dlsym(shared_lib,
        "cudnnSetConvolutionDescriptorEx");
    cudnnDestroyConvolutionDescriptor = (cudnnDestroyConvolutionDescriptor_t)dlsym(shared_lib,
        "cudnnDestroyConvolutionDescriptor");
    cudnnConvolutionForward = (cudnnConvolutionForward_t)dlsym(shared_lib,
        "cudnnConvolutionForward");
    cudnnConvolutionBackwardBias = (cudnnConvolutionBackwardBias_t)dlsym(shared_lib,
        "cudnnConvolutionBackwardBias");
    cudnnConvolutionBackwardData = (cudnnConvolutionBackwardData_t)dlsym(shared_lib,
        "cudnnConvolutionBackwardData");
    cudnnConvolutionBackwardFilter = (cudnnConvolutionBackwardFilter_t)dlsym(shared_lib,
        "cudnnConvolutionBackwardFilter");
  }
  ~cudnn_v1() {
    dlclose(shared_lib);
  }
  static cudnnCreate_t cudnnCreate;
  static cudnnDestroy_t cudnnDestroy;
  static cudnnSetStream_t cudnnSetStream;
  static cudnnGetStream_t cudnnGetStream;
  static cudnnCreateTensor4dDescriptor_t cudnnCreateTensor4dDescriptor;
  static cudnnSetTensor4dDescriptor_t cudnnSetTensor4dDescriptor;
  static cudnnSetTensor4dDescriptorEx_t cudnnSetTensor4dDescriptorEx;
  static cudnnDestroyTensor4dDescriptor_t cudnnDestroyTensor4dDescriptor;
  static cudnnCreateFilterDescriptor_t cudnnCreateFilterDescriptor;
  static cudnnSetFilterDescriptor_t cudnnSetFilterDescriptor;
  static cudnnDestroyFilterDescriptor_t cudnnDestroyFilterDescriptor;
  static cudnnCreateConvolutionDescriptor_t cudnnCreateConvolutionDescriptor;
  static cudnnSetConvolutionDescriptor_t cudnnSetConvolutionDescriptor;
  static cudnnSetConvolutionDescriptorEx_t cudnnSetConvolutionDescriptorEx;
  static cudnnDestroyConvolutionDescriptor_t cudnnDestroyConvolutionDescriptor;
  static cudnnConvolutionForward_t cudnnConvolutionForward;
  static cudnnConvolutionBackwardBias_t cudnnConvolutionBackwardBias;
  static cudnnConvolutionBackwardData_t cudnnConvolutionBackwardData;
  static cudnnConvolutionBackwardFilter_t cudnnConvolutionBackwardFilter;
};

extern cudnn_v1 cudnn_v1_instance;

class CUDNN_V1_ {
 private:
  // disable copy and assignment
  CUDNN_V1_(const CUDNN_V1_&);
  CUDNN_V1_& operator=(const CUDNN_V1_&);
 protected:
  cudnnHandle_t cudnnv1_;
 public:
  explicit CUDNN_V1_() {
    CUDNN_CHECK(cudnn_v1::cudnnCreate(&cudnnv1_));
    CUDNN_CHECK(cudnn_v1::cudnnSetStream(cudnnv1_, stream()));
  }
  virtual ~CUDNN_V1_() {
    CUDNN_CHECK(cudnn_v1::cudnnDestroy(cudnnv1_));
  }

  inline cudnnHandle_t cudnnv1() {
    return cudnnv1_;
  }
};

CUDNN_V1_& cudnn_v1_();

cudnnHandle_t cudnnv1_handle();

}  // namespace purine

#endif
