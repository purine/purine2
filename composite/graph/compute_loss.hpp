// Copyright Lin Min 2015
#ifndef PURINE_COMPUTE_LOSS
#define PURINE_COMPUTE_LOSS

#include <iomanip>
#include <fstream>
#include <set>
#include "composite/composite.hpp"

using namespace std;

namespace purine {

template <typename Net>
class ComputeLoss : public Runnable {
 protected:
  Net* net_;
  vector<Blob*> data_;
  vector<Blob*> labels_;
  vector<Blob*> loss_;
  vector<Blob*> weights_;
 public:
  explicit ComputeLoss(int rank, int device);
  virtual ~ComputeLoss() override {}

  void load(const string& filename);
  void print_loss();
  void feed(const vector<Blob*>& data, const vector<Blob*>& labels);
};

template <typename Net>
ComputeLoss<Net>::ComputeLoss(int rank, int device) : Runnable() {
  net_ = createGraph<Net>("net", rank, device);
  // prune
  const vector<Blob*>& data_diff = net_->data_diff();
  const vector<Blob*>& weight_diff = net_->weight_diff();
  // add to prune
  vector<Node*> to_prune;
  transform(data_diff.begin(), data_diff.end(),
      std::back_inserter(to_prune), [](Blob* b)->Node* {
        return dynamic_cast<Node*>(b);
      });
  transform(weight_diff.begin(), weight_diff.end(),
      std::back_inserter(to_prune), [](Blob* b)->Node* {
        return dynamic_cast<Node*>(b);
      });
  net_->prune(to_prune);

  // get the data and labels
  data_ = net_->data();
  labels_ = net_->label();
  weights_ = net_->weight_data();

  const vector<Blob*>& loss = net_->loss();
  auto copier = createAny<Vectorize<Copy> >("copy_loss",
      vector<Copy::param_tuple>(loss.size(), Copy::param_tuple(0, -1)));
  vector<vector<Blob*> >{ loss } >> *copier;
  loss_ = copier->top()[0];
}

template <typename Net>
void ComputeLoss<Net>::feed(const vector<Blob*>& data,
    const vector<Blob*>& labels) {
  CHECK_EQ(data.size(), data_.size());
  CHECK_EQ(labels.size(), labels_.size());
  for (int i = 0; i < data.size(); ++i) {
    if (current_rank() == data_[i]->rank()) {
      data_[i]->tensor()->swap_memory(data[i]->tensor());
    }
  }
  for (int i = 0; i < labels.size(); ++i) {
    if (current_rank() == labels_[i]->rank()) {
      labels_[i]->tensor()->swap_memory(labels[i]->tensor());
    }
  }
}

template <typename Net>
void ComputeLoss<Net>::print_loss() {
  if (current_rank() == 0) {
    vector<DTYPE> ret(loss_.size());
    transform(loss_.begin(), loss_.end(), ret.begin(), [](Blob* b)->DTYPE {
          return b->tensor()->cpu_data()[0];
        });
    stringstream ss;
    for (int i = 0; i < ret.size(); ++i) {
      ss << "[" << std::scientific << ret[i] << "] ";
    }
    LOG(INFO) << "loss: " << ss.str();
  }
}

template <typename Net>
void ComputeLoss<Net>::load(const string& filename) {
  Runnable loader(0, -1);
  weights_ = net_->weight_data();
  int num_param = weights_.size();
  vector<Blob*> tmp(num_param);
  vector<Blob*> weights(num_param);
  for (int i = 0; i < num_param; ++i) {
    tmp[i] = loader.create("tmp", weights_[i]->tensor()->size());
  }
  for (int i = 0; i < num_param; ++i) {
    weights[i] = loader.create("weight", weights_[i]->shared_tensor());
  }
  vector<vector<Blob*> >{ tmp } >> *loader.createAny<Vectorize<Copy> >("copy",
      vector<Copy::param_tuple>(num_param, Copy::param_tuple())) >>
                                       vector<vector<Blob*> >{ weights };
  if (current_rank() == 0) {
    LOG(INFO) << "Loading snapshot " << filename;
        ifstream in(filename, ios::binary);
    stringstream ss;
    ss << in.rdbuf();
    const string& raw = ss.str();

    int total_len = 0;
    for (Blob* b : tmp) {
      total_len += b->tensor()->size().count() * sizeof(DTYPE);
    }
    CHECK_EQ(raw.length(), total_len) <<
        "Snapshot size incompatible with network weight";
    int offset = 0;
    for (Blob* b : tmp) {
      int len = b->tensor()->size().count() * sizeof(DTYPE);
      memcpy(b->tensor()->mutable_cpu_data(), raw.c_str() + offset, len);
      offset += len;
    }
  }
  loader.run();
  MPI_LOG( << "Snapshot loaded" );
}

}

#endif
