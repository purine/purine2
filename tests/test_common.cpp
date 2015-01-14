// Copyright Lin Min 2014
#include "catch/catch.hpp"

#include <vector>
#include <thread>
#include <chrono>
#include "common/cuda.hpp"

using namespace purine;
using namespace std;

void cudaInThread() {
  CUDA& cu = cuda();
  cudaStream_t stream_ = stream();
  cublasHandle_t cublas_ = cublas_handle();
  curandGenerator_t curand_ = curand_generator();
  cudnnHandle_t cudnn_ = cudnn_handle();
  REQUIRE(cu.stream() == stream_);
  REQUIRE(cu.cublas() == cublas_);
  REQUIRE(cu.curand() == curand_);
  REQUIRE(cu.cudnn() == cudnn_);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  REQUIRE(stream() == stream_);
  REQUIRE(cublas_handle() == cublas_);
  REQUIRE(curand_generator() == curand_);
  REQUIRE(cudnn_handle() == cudnn_);
}

TEST_CASE("TestCUDA", "[CUDA][Thread]") {
  cudaSetDevice(0);
  CUDA& main_cuda = cuda();

  SECTION("new thread") {
    thread t([]() {
          cudaInThread();
        });
    t.join();
  }

  SECTION("diff threads 1") {
    thread t([&main_cuda]() {
          cudaInThread();
          REQUIRE(&main_cuda != &cuda());
          REQUIRE(main_cuda.stream() != stream());
          REQUIRE(main_cuda.cublas() != cublas_handle());
          REQUIRE(main_cuda.curand() != curand_generator());
          REQUIRE(main_cuda.cudnn() != cudnn_handle());
        });
    t.join();
  }

  SECTION("diff threads 2") {
    cudaStream_t stream_;
    cublasHandle_t cublas_;
    curandGenerator_t curand_;
    cudnnHandle_t cudnn_;
    thread t1([&]() {
          cudaInThread();
          stream_ = stream();
          cublas_ = cublas_handle();
          curand_ = curand_generator();
          cudnn_ = cudnn_handle();
          // reserve enough time for the second thread to compare.
          std::this_thread::sleep_for(std::chrono::seconds(2));
        });
    // reserve enough time so that the first thread can finish assignment.
    std::this_thread::sleep_for(std::chrono::seconds(1));
    thread t2([&]() {
          cudaInThread();
          REQUIRE(stream_ != stream());
          REQUIRE(cublas_ != cublas_handle());
          REQUIRE(curand_ != curand_generator());
          REQUIRE(cudnn_ != cudnn_handle());
        });
    t1.join();
    t2.join();
  }

}
