#include "common/cudnn_v1.hpp"

namespace purine {

cudnn_v1::cudnnCreate_t cudnn_v1::cudnnCreate;
cudnn_v1::cudnnDestroy_t cudnn_v1::cudnnDestroy;
cudnn_v1::cudnnSetStream_t cudnn_v1::cudnnSetStream;
cudnn_v1::cudnnGetStream_t cudnn_v1::cudnnGetStream;
cudnn_v1::cudnnCreateTensor4dDescriptor_t cudnn_v1::cudnnCreateTensor4dDescriptor;
cudnn_v1::cudnnSetTensor4dDescriptor_t cudnn_v1::cudnnSetTensor4dDescriptor;
cudnn_v1::cudnnSetTensor4dDescriptorEx_t cudnn_v1::cudnnSetTensor4dDescriptorEx;
cudnn_v1::cudnnDestroyTensor4dDescriptor_t cudnn_v1::cudnnDestroyTensor4dDescriptor;
cudnn_v1::cudnnCreateFilterDescriptor_t cudnn_v1::cudnnCreateFilterDescriptor;
cudnn_v1::cudnnSetFilterDescriptor_t cudnn_v1::cudnnSetFilterDescriptor;
cudnn_v1::cudnnDestroyFilterDescriptor_t cudnn_v1::cudnnDestroyFilterDescriptor;
cudnn_v1::cudnnCreateConvolutionDescriptor_t cudnn_v1::cudnnCreateConvolutionDescriptor;
cudnn_v1::cudnnSetConvolutionDescriptor_t cudnn_v1::cudnnSetConvolutionDescriptor;
cudnn_v1::cudnnSetConvolutionDescriptorEx_t cudnn_v1::cudnnSetConvolutionDescriptorEx;
cudnn_v1::cudnnDestroyConvolutionDescriptor_t cudnn_v1::cudnnDestroyConvolutionDescriptor;
cudnn_v1::cudnnConvolutionForward_t cudnn_v1::cudnnConvolutionForward;
cudnn_v1::cudnnConvolutionBackwardBias_t cudnn_v1::cudnnConvolutionBackwardBias;
cudnn_v1::cudnnConvolutionBackwardData_t cudnn_v1::cudnnConvolutionBackwardData;
cudnn_v1::cudnnConvolutionBackwardFilter_t cudnn_v1::cudnnConvolutionBackwardFilter;

cudnn_v1 cudnn_v1_instance;

CUDNN_V1_& cudnn_v1_() {
  static thread_local CUDNN_V1_ cudnn_v1__;
  return cudnn_v1__;
}

cudnnHandle_t cudnnv1_handle() {
  return cudnn_v1_().cudnnv1();
}

}
