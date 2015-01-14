// Copyright Lin Min 2015
#include <algorithm>
#include <cfloat>
#include <cmath>
#include "operations/include/accuracy.hpp"

namespace purine {

Accuracy::Accuracy(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(topN) = args;
  CHECK_EQ(outputs_[0]->size(), Size(1, 1, 1, 1));
  CHECK_EQ(inputs_[1]->size(), Size(inputs_[0]->size().num(), 1, 1, 1));
}

void Accuracy::compute_cpu(const vector<bool>& add) {
  DTYPE accuracy = 0;
  const DTYPE* bottom_data = inputs_[0]->cpu_data();
  const DTYPE* bottom_label = inputs_[1]->cpu_data();
  Size size = inputs_[0]->size();
  int num = size.num();
  int dim = size.count() / size.num();
  vector<DTYPE> maxval(topN + 1);
  vector<int> max_id(topN + 1);
  for (int i = 0; i < num; ++i) {
    std::vector<std::pair<DTYPE, int> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + j], j));
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + topN,
        bottom_data_vector.end(), std::greater<std::pair<DTYPE, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topN; k++) {
      if (bottom_data_vector[k].second == static_cast<int>(bottom_label[i])) {
        ++accuracy;
        break;
      }
    }
  }
  outputs_[0]->mutable_cpu_data()[0] = accuracy / num;
}

}
