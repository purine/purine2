// Copyright Lin Min 2015
#ifndef PURINE_DATA_PARALLEL
#define PURINE_DATA_PARALLEL

#include <iomanip>
#include <fstream>
#include <set>
#include "composite/composite.hpp"

using namespace std;

namespace purine {

template <typename Net, typename PS>
class DataParallel : public Runnable {
 protected:
  vector<Net*> nets_;
  Vectorize<PS>* param_server_ = NULL;
  vector<Blob*> data_;
  vector<Blob*> labels_;
  vector<Blob*> loss_;
  vector<vector<Blob*> > new_weights_;
  vector<vector<Blob*> > weights_;
 public:
  DataParallel(const vector<pair<int, int> >& locations);
  virtual ~DataParallel() override {};

  // init weight using random number.
  template <typename Random>
  void init(vector<int> index, const typename Random::param_tuple& args);

  /**
   * @brief load weights from snapshot file
   */
  void load(const string& filename);

  /**
   * @brief save weights to snapshot file
   */
  void save(const string& filename);

  vector<DTYPE> loss();
  void print_weight_info();
  void feed(const vector<Blob*>& data, const vector<Blob*>& labels);
  virtual void sync() override;
  PS* param_server(int index) {
    return param_server_->element(index);
  }
  template <typename... Args>
  void setup_param_server(const Args&... args) {
    vector<vector<Blob*> > weight_diff(nets_.size());
    for (int i = 0; i < nets_.size(); ++i) {
      weight_diff[i] = nets_[i]->weight_diff();
    }
    param_server_ = createAny<Vectorize<PS> >("param_server", args...);
    weight_diff >> *param_server_;
    new_weights_ = param_server_->top();
  }
};

template <typename Net, typename PS>
vector<DTYPE> DataParallel<Net, PS>::loss() {
  CHECK_EQ(current_rank(), 0);
  vector<DTYPE> ret(loss_.size());
  transform(loss_.begin(), loss_.end(), ret.begin(), [](Blob* b)->DTYPE {
        return b->tensor()->cpu_data()[0];
      });
  return ret;
}

template <typename Net, typename PS>
void DataParallel<Net, PS>::print_weight_info() {
  if (current_rank() == this->rank_) {
    const vector<Blob*>& weight = nets_[0]->weight_data();
    int max_len = 0;
    for (int i = 0; i < weight.size(); ++i) {
      int len = weight[i]->cached_name().length();
      max_len = max_len > len ? max_len : len;
    }
    for (int i = 0; i < param_server_->size(); ++i) {
      shared_ptr<Tensor> h = param_server_->element(i)->history();
      shared_ptr<Tensor> w = param_server_->element(i)->weight();
      // shared_ptr<Tensor> h = param_server_->element(i)->weight_diff();
      DTYPE h_abs_sum = 0;
      const DTYPE* data = h->cpu_data();
      for (int j = 0; j < h->size().count(); ++j) {
        h_abs_sum += abs(data[j]);
      }
      h_abs_sum /= h->size().count();
      DTYPE w_abs_sum = 0;
      data = w->cpu_data();
      for (int j = 0; j < w->size().count(); ++j) {
        w_abs_sum += abs(data[j]);
      }
      w_abs_sum /= w->size().count();

      const string& name = weight[i]->cached_name();
      size_t pos = name.find("::");
      LOG(INFO) << std::left << std::setw(max_len - pos + 1) <<
          std::setfill(' ') << name.substr(pos + 2) << std::scientific <<
          "(" << w_abs_sum << ") " << " [" << h_abs_sum << "]";
    }
  }
}

template <typename Net, typename PS>
template <typename Random>
void DataParallel<Net, PS>::init(vector<int> index,
    const typename Random::param_tuple& args) {
  Runnable initializer(0, -1);
  Op<Random>* rnd = initializer.create<Random>("init", "main", args);
  vector<Blob*> tmp(index.size());
  vector<vector<Blob*> > weights(nets_.size() + 1);
  for (int i = 0; i < index.size(); ++i) {
    tmp[i] = initializer.create("tmp",
        param_server_->element(index[i])->weight()->size());
  }
  for (int i = 0; i < nets_.size(); ++i) {
    weights[i] = vector<Blob*>(index.size());
    for (int j = 0; j < index.size(); ++j) {
      weights[i][j] = initializer.create("weight",
          nets_[i]->weight_data()[index[j]]->shared_tensor());
    }
  }
  weights[nets_.size()] = vector<Blob*>(index.size());
  for (int j = 0; j < index.size(); ++j) {
    weights[nets_.size()][j] = initializer.create("weight_ps",
        param_server_->element(index[j])->weight());
  }
  vector<vector<Blob*> >{ tmp }
  >> *initializer.createAny<Vectorize<Distribute> >("init_distribute",
      vector<Distribute::param_tuple>(index.size(), Distribute::param_tuple()))
  >> weights;
  *rnd >> tmp;
  initializer.run();
}

template <typename Net, typename PS>
void DataParallel<Net, PS>::load(const string& filename) {
  Runnable loader(0, -1);
  int num_param = param_server_->size();
  vector<Blob*> tmp(num_param);
  vector<vector<Blob*> > weights(nets_.size() + 1);
  for (int i = 0; i < param_server_->size(); ++i) {
    tmp[i] = loader.create("tmp",
        param_server_->element(i)->weight()->size());
  }
  for (int i = 0; i < nets_.size(); ++i) {
    weights[i] = vector<Blob*>(param_server_->size());
    for (int j = 0; j < param_server_->size(); ++j) {
      weights[i][j] = loader.create("weight",
          nets_[i]->weight_data()[j]->shared_tensor());
    }
  }
  weights[nets_.size()] = vector<Blob*>(param_server_->size());
  for (int j = 0; j < param_server_->size(); ++j) {
    weights[nets_.size()][j] = loader.create("weight_ps",
        param_server_->element(j)->weight());
  }
  vector<vector<Blob*> >{ tmp }
  >> *loader.createAny<Vectorize<Distribute> >("init_distribute",
      vector<Distribute::param_tuple>(param_server_->size(),
          Distribute::param_tuple()))
  >> weights;
  // fill with the binary data
  if (current_rank() == 0) {
    LOG(INFO) << "Loading snapshot " << filename;
    // read file into binary string raw
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
  // run
  loader.run();
  MPI_LOG( << "Snapshot loaded" );
}

template <typename Net, typename PS>
void DataParallel<Net, PS>::save(const string& filename) {
  Runnable saver;
  int param_num = param_server_->size();
  vector<Blob*> param(param_num);
  for (int i = 0; i < param_num; ++i) {
    param[i] = saver.create("param", param_server_->element(i)->weight());
  }
  auto copier = saver.createAny<Vectorize<Copy> >("copy_here",
      vector<Copy::param_tuple>(param_num, Copy::param_tuple(0, -1)));
  vector<vector<Blob*> >{ param } >> *copier;
  vector<Blob*> copied = copier->top()[0];
  saver.run();

  if (current_rank() == 0) {
    ofstream out(filename);
    for (int i = 0; i < param_num; ++i) {
      const char* data = reinterpret_cast<const char*>(
          copied[i]->tensor()->cpu_data());
      int len = copied[i]->tensor()->size().count() * sizeof(DTYPE);
      out.write(data, len);
    }
    LOG(INFO) << "Saving snapshot " << filename;
  }
}

template <typename Net, typename PS>
DataParallel<Net, PS>::DataParallel(const vector<pair<int, int> >& locations)
    : Runnable() {
  // create replica
  vector<vector<Blob*> > losses;
  nets_ = vector<Net*>(locations.size());
  weights_ = vector<vector<Blob*> >(locations.size());
  for (int i = 0; i < locations.size(); ++i) {
    nets_[i] = createGraph<Net>("replica" + to_string(i), locations[i].first,
        locations[i].second);
    const vector<Blob*>& data_diff = nets_[i]->data_diff();
    vector<Node*> to_prune(data_diff.size());
    transform(data_diff.begin(), data_diff.end(), to_prune.begin(),
        [](Blob* b)->Node* {
          return dynamic_cast<Node*>(b);
        });
    nets_[i]->prune(to_prune);
    // get the data and labels
    const vector<Blob*>& dt = nets_[i]->data();
    const vector<Blob*>& lb = nets_[i]->label();
    data_.insert(data_.end(), dt.begin(), dt.end());
    labels_.insert(labels_.end(), lb.begin(), lb.end());
    losses.push_back(nets_[i]->loss());
    weights_[i] = nets_[i]->weight_data();
  }
  // agg loss to rank 0 device -1.
  Vectorize<Aggregate>* agg = createAny<Vectorize<Aggregate> >("agg_loss",
      vector<Aggregate::param_tuple>(losses[0].size(),
          Aggregate::param_tuple(Aggregate::AVERAGE, 0, -1)));
  losses >> *agg;
  loss_ = agg->top()[0];
}

template <typename Net, typename PS>
void DataParallel<Net, PS>::feed(const vector<Blob*>& data,
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

template <typename Net, typename PS>
void DataParallel<Net, PS>::sync() {
  Runnable::sync();
  // update the weights
  for (int i = 0; i < nets_.size(); ++i) {
    if (nets_[i]->rank() == current_rank()) {
      for (int j = 0; j < weights_[0].size(); ++j) {
        CHECK_EQ(new_weights_[i][j]->tensor()->size(),
            weights_[i][j]->tensor()->size());
        CHECK_EQ(new_weights_[i][j]->rank(), weights_[i][j]->rank());
        CHECK_EQ(new_weights_[i][j]->device(), weights_[i][j]->device());
        new_weights_[i][j]->tensor()->swap_memory(weights_[i][j]->tensor());
      }
    }
  }
}

}

#endif
