// Copyright Lin Min 2015
#ifndef PURINE_MEM_COPY
#define PURINE_MEM_COPY

#include "operations/operation.hpp"

namespace purine {

class MemCopy : public Operation {
 // protected:
 //  cudnnTensorDescriptor_t bottom_desc_ = nullptr;
 //  cudnnTensorDescriptor_t top_desc_ = nullptr;
 public:
  typedef tuple<> param_tuple;
  explicit MemCopy(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
