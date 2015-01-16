#ifndef PURINE_CUDA
#define PURINE_CUDA

#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <cudnn.h>
#include <driver_types.h>  // cuda driver types

#include "common/common.hpp"

namespace purine {

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << curandGetErrorString(status); \
  } while (0)

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " " \
      << cudnnGetErrorString(status); \
  } while (0)

/**
 * @def THREAD_SET_CUDA_DEVICE(device_id)
 * @brief set the cuda device of the thread.
 *
 * @param device_id is the device ordinal.
 */
#define THREAD_SET_CUDA_DEVICE(device_id) \
  int device_count = 0; \
  CUDA_CHECK(cudaGetDeviceCount(&device_count)); \
  CHECK_LT(device_id, device_count); \
  CUDA_CHECK(cudaSetDevice(device_id))

/**
 * set cuda device to device_id and keep the original device_id
 * need to call switch back as a pair
 */
#define SWITCH_DEVICE(device_id) \
  int current_device_id_; \
  CUDA_CHECK(cudaGetDevice(&current_device_id_)); \
  if (device_id != current_device_id_) { \
    CUDA_CHECK(cudaSetDevice(device_id)); \
  }
#define SWITCH_BACK(device_id) \
  if (current_device_id_ != device_id) { \
    CUDA_CHECK(cudaSetDevice(current_device_id_)); \
  }

/**
 * @brief return error string of cublasStatus_t
 */
const char* cublasGetErrorString(cublasStatus_t error);

/**
 * @brief return error string of curandStatus_t
 */
const char* curandGetErrorString(curandStatus_t error);

class CUDA {
 private:
  // disable copy and assignment
  CUDA(const CUDA&);
  CUDA& operator=(const CUDA&);
 protected:
  cudaStream_t stream_;
  cublasHandle_t cublas_;
  curandGenerator_t curand_;
  cudnnHandle_t cudnn_;
 public:
  explicit CUDA() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CURAND_CHECK(curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
        curand_, cluster_seedgen()));
    CUDNN_CHECK(cudnnCreate(&cudnn_));
    // set cuda stream to the handles
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    CURAND_CHECK(curandSetStream(curand_, stream_));
    CUDNN_CHECK(cudnnSetStream(cudnn_, stream_));
  }
  virtual ~CUDA() {
    CUDA_CHECK(cudaStreamDestroy(stream_));
    CUBLAS_CHECK(cublasDestroy(cublas_));
    CURAND_CHECK(curandDestroyGenerator(curand_));
    CUDNN_CHECK(cudnnDestroy(cudnn_));
  }

  inline cudaStream_t stream() {
    return stream_;
  }

  inline cublasHandle_t cublas() {
    return cublas_;
  }

  inline curandGenerator_t curand() {
    return curand_;
  }

  inline cudnnHandle_t cudnn() {
    return cudnn_;
  }

};

/**
 * @fn CUDA& cuda()
 * @brief cuda return a thread_local and static instance of CUDA.
 *        which contains thread specific cuda handles.
 */
CUDA& cuda();

/**
 * @fn cudaStream_t stream()
 * @brief stream returns a thread specific cudaStream_t handle to a cuda stream.
 *
 * the handle will come into being at the first call of stream() in the thread
 * and be destroyed when the thread exits.
 */
cudaStream_t stream();

/**
 * @fn cudaHandle_t cublas_handle()
 * @brief cublas_handle returns a thread specific
 *        cublasHandle_t handle to cublas.
 *
 * the handle will come into being at the first call of
 * cublas_handle() in the thread and be destroyed when the thread exits.
 */
cublasHandle_t cublas_handle();

/**
 * @fn curandGenerator_t curand_generator()
 * @brief curand_generator returns a thread specific
 *        curandGenerator_t handle to curandGenerator.
 *
 * the handle will come into being at the first call of
 * curand_generator() in the thread and be destroyed when the thread exits.
 */
curandGenerator_t curand_generator();

/**
 * @fn cudnnHandle_t cudnn_handle()
 * @brief cudnn_handle returns a thread specific
 *        cudnnHandle_t handle to cudnn.
 *
 * the handle will come into being at the first call of
 * cudnn_handle() in the thread and be destroyed when the thread exits.
 */
cudnnHandle_t cudnn_handle();


}  // namespace purine

#endif
