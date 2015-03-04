// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/nin_cifar10.hpp"
#include "composite/graph/all_reduce.hpp"

int batch_size = 128;
string source = "/temp/cifar-train-lmdb";
string mean_file = "";

using namespace purine;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // initilize MPI
  int ret;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
  // parallels
  vector<pair<int, int> > parallels;
  for (int rank : {0}) {
    for (int device : {0}) {
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
  shared_ptr<DataParallel<NIN_Cifar10<false>, AllReduce> > parallel_nin_cifar
      = make_shared<DataParallel<NIN_Cifar10<false>, AllReduce> >(parallels);
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
  for (int iter = 1; iter <= 50000; ++iter) {
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
