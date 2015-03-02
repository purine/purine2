// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/googlenet.hpp"
#include "composite/graph/all_reduce.hpp"

int batch_size = 128;
string source = "/temp/imagenet-train-256xN-lmdb";
string mean_file = "/temp/imagenet-train-mean";

using namespace purine;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // initilize MPI
  int ret;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
  // parallels
  vector<pair<int, int> > parallels;
  for (int rank : {0, 1, 2, 3}) {
    for (int device : {0, 1, 2}) {
      parallels.push_back({rank, device});
    }
  }
  // parameter server
  pair<int, int> param_server = {0, -1};
  // fetch image
  shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
      true, true, true, batch_size, 224, parallels);
  fetch->run();
  // create data parallelism of GoogLeNet;
  shared_ptr<DataParallel<GoogLeNet, AllReduce> > parallel_googlenet
      = make_shared<DataParallel<GoogLeNet, AllReduce> >(parallels);
  // set learning rate etc
  DTYPE global_learning_rate = 0.05;
  DTYPE global_decay = 0.0005;
  vector<AllReduce::param_tuple> param(116);
  for (int i = 0; i < 116; ++i) {
    DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
    param[i] = AllReduce::param_tuple(0.9, learning_rate,
        learning_rate * global_decay * (i % 2 ? 0. : 1.));
  }
  parallel_googlenet->setup_param_server(vector<int>(116, 0),
      vector<int>(116, -1), param);
  // do the initialization
#define RANDOM
#ifdef RANDOM
  vector<int> indice(58);
  iota(indice.begin(), indice.end(), 0);
  vector<int> weight_indice(58);
  vector<int> bias_indice(58);
  transform(indice.begin(), indice.end(), weight_indice.begin(),
      [](int i)->int {
        return i * 2;
      });
  transform(indice.begin(), indice.end(), bias_indice.begin(),
      [](int i)->int {
        return i * 2 + 1;
      });
  parallel_googlenet->init<Constant>(bias_indice, Constant::param_tuple(0.));
  parallel_googlenet->init<Gaussian>(weight_indice,
      Gaussian::param_tuple(0., 0.05));
  parallel_googlenet->init<Gaussian>({0, 4, 114, 110, 106, 98, 94},
      Gaussian::param_tuple(0., 0.01));
#else
  parallel_googlenet->load("./googlenet_no_aux_dump_iter_25000.snapshot");
#endif
  // iteration
  for (int iter = 1; iter <= 35000; ++iter) {
    // feed prefetched data to googlenet
    parallel_googlenet->feed(fetch->images(), fetch->labels());
    // start googlenet and next fetch
    parallel_googlenet->run_async();
    fetch->run_async();
    fetch->sync();
    parallel_googlenet->sync();
    // verbose
    MPI_LOG( << "iteration: " << iter << ", loss: "
        << parallel_googlenet->loss()[0]);
    if (iter % 100 == 0) {
      parallel_googlenet->print_weight_info();
    }
    if (iter % 5000 == 0) {
      parallel_googlenet->save("./googlenet_no_aux_dump_iter_"
          + to_string(iter) + ".snapshot");
    }
  }
  // delete
  fetch.reset();
  parallel_googlenet.reset();
  // Finalize MPI
  MPI_CHECK(MPI_Finalize());
  return 0;
}
