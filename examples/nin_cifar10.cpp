// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "common/common.hpp"
#include "dispatch/runnable.hpp"
#include "composite/composite.hpp"
#include "composite/graph/all_reduce.hpp"

namespace purine {

int batch_size = 128;
string source = "/temp/cifar-train-lmdb";
string mean_file = "";

class NIN_Cifar10 : public Graph {
 protected:
  Blob* data_;
  Blob* label_;
  Blob* data_diff_;
  vector<Blob*> weights_;
  vector<Blob*> weight_data_;
  vector<Blob*> weight_diff_;
  vector<Blob*> loss_;
 public:
  explicit NIN_Cifar10(int rank, int device);
  virtual ~NIN_Cifar10() override {}
  inline const vector<Blob*>& weight_data() { return weight_data_; }
  inline const vector<Blob*>& weight_diff() { return weight_diff_; }
  inline vector<Blob*> data() { return { data_ }; }
  inline vector<Blob*> label() { return { label_ }; }
  inline vector<Blob*> data_diff() { return { data_diff_ }; }
  inline vector<Blob*> loss() { return loss_; }
};

NIN_Cifar10::NIN_Cifar10(int rank, int device) : Graph(rank, device) {
  data_ = create("data", { batch_size, 3, 32, 32 });
  data_diff_ = create("data_diff", { batch_size, 3, 32, 32 });
  label_ = create("label", { batch_size, 1, 1, 1 });

  // creating layers
  NINLayer* nin1 = createGraph<NINLayer>("nin1",
      NINLayer::param_tuple(2, 2, 1, 1, 5, 5, "relu", {192, 160, 96}));
  PoolLayer* pool1 = createGraph<PoolLayer>("pool1",
      PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  DropoutLayer* dropout1 = createGraph<DropoutLayer>("dropout1",
      DropoutLayer::param_tuple(0.5, false));

  NINLayer* nin2 = createGraph<NINLayer>("nin2",
      NINLayer::param_tuple(2, 2, 1, 1, 5, 5, "relu", {192, 192, 192}));
  PoolLayer* pool2 = createGraph<PoolLayer>("pool2",
      PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  DropoutLayer* dropout2 = createGraph<DropoutLayer>("dropout2",
      DropoutLayer::param_tuple(0.5, false));

  NINLayer* nin3 = createGraph<NINLayer>("nin3",
      NINLayer::param_tuple(1, 1, 1, 1, 3, 3, "relu", {192, 192, 10}));

  GlobalAverageLayer* global_ave = createGraph<GlobalAverageLayer>("global_avg",
      GlobalAverageLayer::param_tuple());
  SoftmaxLossLayer* softmaxloss = createGraph<SoftmaxLossLayer>("softmaxloss",
      SoftmaxLossLayer::param_tuple(1.));

  // connecting layers
  B{ data_,  data_diff_ } >> *nin1 >> *pool1 >> *dropout1
  >> *nin2 >> *pool2 >> *dropout2 >> *nin3 >> *global_ave;
  // loss layer
  softmaxloss->set_label(label_);
  *global_ave >> *softmaxloss;
  // loss
  loss_ = { softmaxloss->loss()[0] };
  // weight
  vector<Layer*> layers = { nin1, nin2, nin3 };
  for (auto layer : layers) {
    const vector<Blob*>& w = layer->weight_data();
    weight_data_.insert(weight_data_.end(), w.begin(), w.end());
  }
  for (auto layer : layers) {
    const vector<Blob*>& w = layer->weight_diff();
    weight_diff_.insert(weight_diff_.end(), w.begin(), w.end());
  }
}

}

using namespace purine;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // initilize MPI
  int ret;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
  // parallels
  vector<pair<int, int> > parallels;
  for (int rank : {0}) {
    for (int device : {0, 1, 2}) {
      parallels.push_back({rank, device});
    }
  }
  // parameter server
  pair<int, int> param_server = {0, -1};
  // fetch image
  shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
      false, false, true, batch_size, 32, parallels);
  fetch->run();
  // create data parallelism of Nin_Cifar;
  shared_ptr<DataParallel<NIN_Cifar10, AllReduce> > parallel_nin_cifar
      = make_shared<DataParallel<NIN_Cifar10, AllReduce> >(parallels);
  // set learning rate etc
  DTYPE global_learning_rate = 0.1;
  DTYPE global_decay = 0.0001;
  vector<AllReduce::param_tuple> param(18);
  for (int i = 0; i < 18; ++i) {
    DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
    if (i == 16 || i == 17) {
      learning_rate /= 10.;
    }
    param[i] = AllReduce::param_tuple(0.9, learning_rate,
        learning_rate * global_decay * (i % 2 ? 0. : 1.));
  }
  parallel_nin_cifar->setup_param_server(vector<int>(18, 0),
      vector<int>(18, -1), param);
  // do the initialization
#define RANDOM
#ifdef RANDOM
  vector<int> indice(9);
  iota(indice.begin(), indice.end(), 0);
  vector<int> weight_indice(9);
  vector<int> bias_indice(9);
  transform(indice.begin(), indice.end(), weight_indice.begin(),
      [](int i)->int {
        return i * 2;
      });
  transform(indice.begin(), indice.end(), bias_indice.begin(),
      [](int i)->int {
        return i * 2 + 1;
      });
  parallel_nin_cifar->init<Constant>(bias_indice, Constant::param_tuple(0.));
  parallel_nin_cifar->init<Gaussian>(weight_indice,
      Gaussian::param_tuple(0., 0.05));
#else
  parallel_nin_cifar->load("./nin_cifar_dump_iter_25000.snapshot");
#endif
  // iteration
  for (int iter = 1; iter <= 35000; ++iter) {
    // feed prefetched data to nin_cifar
    parallel_nin_cifar->feed(fetch->images(), fetch->labels());
    // start nin_cifar and next fetch
    parallel_nin_cifar->run_async();
    fetch->run_async();
    fetch->sync();
    parallel_nin_cifar->sync();
    // verbose
    MPI_LOG( << "iteration: " << iter << ", loss: "
        << parallel_nin_cifar->loss()[0]);
    if (iter % 100 == 0) {
      parallel_nin_cifar->print_weight_info();
    }
    if (iter % 5000 == 0) {
      parallel_nin_cifar->save("./nin_cifar_dump_iter_"
          + to_string(iter) + ".snapshot");
    }
  }
  // delete
  fetch.reset();
  parallel_nin_cifar.reset();
  // Finalize MPI
  MPI_CHECK(MPI_Finalize());
  return 0;
}
