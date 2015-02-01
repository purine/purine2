// Copyright Lin Min 2015
#ifndef PURINE_IMAGE_LABEL
#define PURINE_IMAGE_LABEL

#include <lmdb.h>
#include <memory>

#include "operations/operation.hpp"
#include "caffeine/math_functions.hpp"

namespace purine {

/**
 * {} >> op >> { image, label }
 */
class ImageLabel : public Operation {
 protected:
  string source;
  string mean;
  bool mirror;
  bool random;
  bool color;
  int interval;
  int offset;
  int batch_size;
  int crop_size;

  shared_ptr<Tensor> mean_;
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

 public:
  typedef tuple<string, string, bool, bool, bool,
                int, int, int, int> param_tuple;
  explicit ImageLabel(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  ~ImageLabel();
  virtual void compute_cpu(const vector<bool>& add);
};

}

#endif
