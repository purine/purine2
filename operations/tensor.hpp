#ifndef PURINE_TENSOR
#define PURINE_TENSOR

#include <memory>

#include "common/common.hpp"
#include "common/cuda.hpp"
#include "operations/size.hpp"

using std::shared_ptr;

namespace purine {

class Tensor {
 public:
  explicit Tensor(int rank, int device, const Size& size,
      const Offset& offset, const Stride& stride);
  explicit Tensor(int rank, int device, const Size& size);
  virtual ~Tensor();

  inline const Size& size() const { return size_; }
  inline const Stride& stride() const { return stride_; }
  inline const Offset& offset() const { return offset_; }
  inline int rank() const { return rank_; }
  inline int device() const { return device_; }

  void swap_memory(Tensor* other);
  void share_from(Tensor* other);
  void slice_from(Tensor* other, const Offset& off, const Size& size);
  void delete_data();

  inline DTYPE* mutable_gpu_data() {
    CHECK(device_ >= 0);
    return mutable_data();
  }
  inline const DTYPE* gpu_data() {
    CHECK(device_ >= 0);
    return data();
  }
  inline DTYPE* mutable_cpu_data() {
    CHECK(device_ < 0);
    return mutable_data();
  }
  inline const DTYPE* cpu_data() {
    CHECK(device_ < 0);
    return data();
  }
  DTYPE* mutable_data();
  const DTYPE* data() const;

  bool is_contiguous() const;

 protected:
  Size size_;
  Offset offset_;
  Stride stride_;
  shared_ptr<DTYPE> data_;
  int rank_;
  int device_;
  // static
  static int offset(const Offset& off, const Stride& stride);
  static void alloc_mem(DTYPE** data, const Size& size, int rank, int device);
  static void free_mem(DTYPE* data, int rank, int device);
};

}

#endif
