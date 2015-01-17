#ifndef PURINE_BLOB
#define PURINE_BLOB

#include <memory>

#include "dispatch/node.hpp"
#include "operations/operation.hpp"
#include "operations/tensor.hpp"

namespace purine {

using std::shared_ptr;

template <typename O> class Op;

class Blob : public Node {
  template <typename O>
  friend Op<O>& operator >> (const vector<Blob*>& inputs, Op<O>& op);
  template <typename O>
  friend void operator >> (Op<O>& op, const vector<Blob*>& outputs);
 protected:
  shared_ptr<Tensor> tensor_;
  cudaEvent_t cuda_event_ = NULL;
 public:
  explicit Blob(const Size& s, int rank, int device);
  virtual ~Blob();
  inline cudaEvent_t cuda_event() { return cuda_event_; }
  Tensor* tensor();
  virtual void run();
};

}

#endif
