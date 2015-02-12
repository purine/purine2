// Copyright Lin Min 2015
#ifndef PURINE_DATA_PARALLEL
#define PURINE_DATA_PARALLEL

#include <iomanip>
#include "composite/composite.hpp"

using namespace std;

namespace purine {

template <typename Net>
class DataParallel : public Runnable {
 protected:
  vector<Net*> nets_;
  vector<Blob*> weight_;
  vector<Blob*> new_weight_;
  vector<Blob*> history_;
  vector<Blob*> new_history_;
  vector<vector<Blob*> > nets_weight_data_;
  vector<vector<Blob*> > nets_weight_diff_;
  vector<vector<Blob*> > new_weights_;
  shared_ptr<Runnable> compute_loss_;
  vector<Blob*> loss_;
  vector<Blob*> data_;
  vector<Blob*> labels_;
 public:
  DataParallel(const vector<pair<int, int> >& locations,
      const pair<int, int>& param_server);
  virtual ~DataParallel() override {};
  template <typename Random>
  void init(vector<int> index, const typename Random::param_tuple& args);
  vector<DTYPE> loss();
  void print_weight_diff_l1();
  void print_weight_l1();
  void feed(const vector<Blob*>& data, const vector<Blob*>& labels);
  virtual void run() override;
};

template <typename Net>
vector<DTYPE> DataParallel<Net>::loss() {
  if (!compute_loss_) {
    // compute_loss
    int loss_count = nets_[0]->loss().size();
    compute_loss_.reset(new Runnable(rank_, device_));
    loss_ = vector<Blob*>(loss_count);
    for_each(loss_.begin(), loss_.end(), [this](Blob*& blob){
          blob = compute_loss_->create("[loss]", rank_, -1, {1, 1, 1, 1});
        });
    vector<vector<Blob*> > net_losses(nets_.size());
    for (int i = 0; i < nets_.size(); ++i) {
      const vector<Blob*>& l = nets_[i]->loss();
      net_losses[i] = vector<Blob*>(loss_count);
      for (int j = 0; j < l.size(); ++j) {
        net_losses[i][j] = compute_loss_->create("loss",
            l[j]->shared_tensor());
      }
    }
    net_losses >> *(compute_loss_->createAny<Vectorize<Aggregate> >(
        "agg_loss", vector<Aggregate::param_tuple>(loss_count,
            Aggregate::param_tuple(Aggregate::AVERAGE, rank_, device_))))
    >> vector<vector<Blob*> >{ loss_ };
  }
  // run the compute loss graph
  compute_loss_->run();
  vector<DTYPE> ret(loss_.size());
  transform(loss_.begin(), loss_.end(), ret.begin(), [](Blob* b)->DTYPE {
        return b->tensor()->cpu_data()[0];
      });
  return ret;
}

template <typename Net>
void DataParallel<Net>::print_weight_diff_l1() {
  if (current_rank() == this->rank_) {
    int max_len = 0;
    for (int i = 0; i < history_.size(); ++i) {
      int len = nets_weight_data_[0][i]->cached_name().length();
      max_len = max_len > len ? max_len : len;
    }
    for (int i = 0; i < history_.size(); ++i) {
      Blob* b = history_[i];
      DTYPE abs_sum = 0;
      const DTYPE* data = b->tensor()->cpu_data();
      for (int i = 0; i < b->tensor()->size().count(); ++i) {
        abs_sum += abs(data[i]);
      }
      abs_sum /= b->tensor()->size().count();
      const string& name = nets_weight_data_[0][i]->cached_name();
      size_t pos = name.find("::");
      LOG(INFO) << std::left << std::setw(max_len - pos + 1) <<
          std::setfill(' ') << name.substr(pos + 2) << std::scientific <<
          "[" << abs_sum << "]";
    }
  }
}

template <typename Net>
void DataParallel<Net>::print_weight_l1() {
  if (current_rank() == this->rank_) {
    for (int i = 0; i < weight_.size(); ++i) {
      Blob* b = weight_[i];
      DTYPE abs_sum = 0;
      const DTYPE* data = b->tensor()->cpu_data();
      for (int i = 0; i < b->tensor()->size().count(); ++i) {
        abs_sum += abs(data[i]);
      }
      abs_sum /= b->tensor()->size().count();
      LOG(INFO) << nets_weight_data_[0][i]->cached_name()
      << "(" << abs_sum << ")";
    }
  }
}

template <typename Net>
template <typename Random>
void DataParallel<Net>::init(vector<int> index,
    const typename Random::param_tuple& args) {
  Runnable initializer(rank_, device_);
  Op<Random>* rnd = initializer.create<Random>("init", "main", args);
  vector<Blob*> tmp(index.size());
  vector<vector<Blob*> > weights(nets_.size());
  for (int i = 0; i < index.size(); ++i) {
    tmp[i] = initializer.create("tmp", weight_[index[i]]->shared_tensor());
  }
  for (int i = 0; i < nets_weight_data_.size(); ++i) {
    weights[i] = vector<Blob*>(index.size());
    for (int j = 0; j < index.size(); ++j) {
      weights[i][j] = initializer.create("weight",
          nets_weight_data_[i][index[j]]->shared_tensor());
    }
  }
  vector<vector<Blob*> >{ tmp }
  >> *initializer.createAny<Vectorize<Distribute> >("init_distribute",
      vector<Distribute::param_tuple>(index.size(), Distribute::param_tuple()))
  >> weights;
  *rnd >> tmp;
  initializer.run();
}

template <typename Net>
DataParallel<Net>::DataParallel(const vector<pair<int, int> >& locations,
    const pair<int, int>& param_server) : Runnable(param_server.first,
        param_server.second) {
  // create replica
  nets_ = vector<Net*>(locations.size());
  for (int i = 0; i < locations.size(); ++i) {
    nets_[i] = createGraph<Net>("replica", locations[i].first,
        locations[i].second);
    // get the data and labels
    const vector<Blob*>& dt = nets_[i]->data();
    const vector<Blob*>& lb = nets_[i]->label();
    data_.insert(data_.end(), dt.begin(), dt.end());
    labels_.insert(labels_.end(), lb.begin(), lb.end());
  }

  // weights and weight_diffs
  nets_weight_data_ = vector<vector<Blob*> >(nets_.size());
  nets_weight_diff_ = vector<vector<Blob*> >(nets_.size());
  new_weights_ = vector<vector<Blob*> >(nets_.size());
  for (int i = 0; i < nets_.size(); ++i) {
    nets_weight_data_[i] = nets_[i]->weight_data();
    nets_weight_diff_[i] = nets_[i]->weight_diff();

    new_weights_[i] = nets_[i]->weight_data();
    for (int j = 0; j < new_weights_[i].size(); ++j) {
      new_weights_[i][j] = create("new_weight_local",
          nets_weight_data_[i][j]->rank(), nets_weight_data_[i][j]->device(),
          nets_weight_data_[i][j]->tensor()->size());
    }
  }
  int param_num = nets_weight_data_[0].size();

  // aggregate weight_diff
  Vectorize<Aggregate>* agg = createAny<Vectorize<Aggregate> >("aggregate",
      vector<Aggregate::param_tuple>(param_num,
          Aggregate::param_tuple(Aggregate::AVERAGE, rank_, device_)));
  nets_weight_diff_ >> *agg;

  // create weight, history, new_weight, new_history
  weight_ = vector<Blob*>(param_num);
  history_ = vector<Blob*>(param_num);
  new_weight_ = vector<Blob*>(param_num);
  new_history_ = vector<Blob*>(param_num);
  for (int i = 0; i < param_num; ++i) {
    Size weight_size = nets_weight_data_[0][i]->tensor()->size();
    weight_[i] = create("weight", weight_size);
    history_[i] = create("history", weight_size);
    new_weight_[i] = create("new_weight", weight_[i]->shared_tensor());
    new_history_[i] = create("new_history", history_[i]->shared_tensor());
  }

  // update
  vector<Update::param_tuple> update_params();
  Vectorize<Update>* updator = createAny<Vectorize<Update> >(
      "updator", vector<int>(param_num, rank_), vector<int>(param_num, device_),
      vector<Update::param_tuple>(param_num,
          Update::param_tuple(0.9, 0.01, 0.0005)));
  vector<vector<Blob*> >{ weight_, agg->top()[0], history_ }
  >> *updator >> vector<vector<Blob*> >{ new_weight_, new_history_ };

  // distribute weight
  vector<vector<Blob*> >{ new_weight_ }
  >> *createAny<Vectorize<Distribute> >("distribute",
      vector<Distribute::param_tuple>(param_num, Distribute::param_tuple()))
         >> new_weights_;

  // initilize history
  Runnable fill_history(rank_, device_);
  vector<Blob*> h(history_.size());
  for (int i = 0; i < history_.size(); ++i) {
    h[i] = fill_history.create("history", history_[i]->shared_tensor());
  }
  *fill_history.create<Constant>("fill_history", "main",
      Constant::param_tuple(0.)) >> h;
  fill_history.run();
}

template <typename Net>
void DataParallel<Net>::feed(const vector<Blob*>& data,
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
void DataParallel<Net>::run() {
  // run and sync
  run_async();
  sync();
  // update the weights
  for (int i = 0; i < nets_.size(); ++i) {
    if (nets_[i]->rank() == current_rank()) {
      for (int j = 0; j < nets_.size(); ++j) {
        new_weights_[i][j]->tensor()->
            swap_memory(nets_weight_data_[i][j]->tensor());
      }
    }
  }
}

}

#endif
