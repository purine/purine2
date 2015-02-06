// Copyright Lin Min 2015
#include "composite/graph/update.hpp"
#include "operations/include/dummy.hpp"
namespace purine {

void Update::setup() {
  CHECK(bottom_setup_);
  CHECK_EQ(bottom_.size(), 3);
  Size bottom_size = bottom_[0]->tensor()->size();
  // check top
  if (top_.size() != 0) {
    CHECK_EQ(top_.size(), 2);
  } else {
    top_ = {
      create("new_weight", bottom_size),
      create("new_history", bottom_size)
    };
  }

  // 'update' shares tensor from new_history
  Blob* update = create("update", top_[1]->shared_tensor());
  // create ops
  Op<WeightedSum>* compute_update = create<WeightedSum>("compute_update",
      "main",
      WeightedSum::param_tuple({momentum, learning_rate, weight_decay}));

  Op<WeightedSum>* apply_update = create<WeightedSum>("apply_update", "main",
      WeightedSum::param_tuple({1., -1.}));

  vector<Blob*>{ bottom_[2], bottom_[1], bottom_[0] } >> *compute_update
               >> vector<Blob*>{ update };
  vector<Blob*>{ bottom_[0], update } >> *apply_update >>
                                             vector<Blob*>{ top_[0] };
  vector<Blob*>{ update } >>
      *create<Dummy>("dummy", "main", Dummy::param_tuple()) >>
          vector<Blob*>{ top_[1] };
}

}
