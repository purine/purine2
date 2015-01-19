#ifndef PURINE_BLOB
#define PURINE_BLOB

#include <memory>

#include "dispatch/node.hpp"
#include "operations/operation.hpp"
#include "operations/tensor.hpp"

namespace purine {

using std::shared_ptr;

class Op_;

class Blob : public Node {
  friend Op_& operator >> (const vector<Blob*>& inputs, Op_& op);
  friend void operator >> (Op_& op, const vector<Blob*>& outputs);
 protected:
  shared_ptr<Tensor> tensor_;
  cudaEvent_t cuda_event_ = NULL;
 public:
  explicit Blob(const Size& s, int rank = 0, int device = 0);
  Blob(shared_ptr<Tensor> tensor);
  virtual ~Blob();
  inline cudaEvent_t cuda_event() { return cuda_event_; }
  Tensor* tensor();
  shared_ptr<Tensor> shared_tensor();
  virtual void run();
};

}

#endif
