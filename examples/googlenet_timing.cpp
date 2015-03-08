// Copyright Lin Min 2015

#include <mpi.h>
#include <chrono>
#include <glog/logging.h>
#include "examples/googlenet.hpp"
#include "composite/graph/all_reduce.hpp"

int batch_size = 128;
string source = "/temp/imagenet-train-256xN-lmdb";
string mean_file = "/temp/imagenet-train-mean";

using namespace purine;

void setup_param_server(DataParallel<GoogLeNet<false>, AllReduce>*
    parallel_googlenet, DTYPE global_learning_rate) {
  // set learning rate etc
  DTYPE global_decay = 0.0001;
  vector<AllReduce::param_tuple> param(116);
  for (int i = 0; i < 116; ++i) {
    DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
    param[i] = AllReduce::param_tuple(0.9, learning_rate,
        learning_rate * global_decay * (i % 2 ? 0. : 1.));
  }
  parallel_googlenet->setup_param_server(vector<int>(116, 0),
      vector<int>(116, -1), param);
}

void initialize(DataParallel<GoogLeNet<false>, AllReduce>* parallel_googlenet,
                const string& snapshot) {
  if (snapshot == "") {
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
    parallel_googlenet->init<Constant>(bias_indice, Constant::param_tuple(0.2));
    parallel_googlenet->init<Gaussian>(weight_indice,
        Gaussian::param_tuple(0., 0.05));
    parallel_googlenet->init<Gaussian>({0, 2,
            12, 14,
            24, 26,
            36, 38,
            48, 50,
            60, 62,
            72, 74,
            84, 86,
            96, 98,
            108, 110,
            114},
        Gaussian::param_tuple(0., 0.01));
  } else {
    parallel_googlenet->load(snapshot);
  }
}

void time(int bs) {
  batch_size = bs;
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
  // googlenet
  shared_ptr<DataParallel<GoogLeNet<false>, AllReduce> > parallel_googlenet
      = make_shared<DataParallel<GoogLeNet<false>, AllReduce> >(parallels);
  setup_param_server(parallel_googlenet.get(), 0.1);
  initialize(parallel_googlenet.get(), "");

  auto start = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
          (std::chrono::system_clock::now() - start);
  // iteration
  for (int iter = 1; iter <= 200; ++iter) {
    // feed prefetched data to googlenet
    parallel_googlenet->feed(fetch->images(), fetch->labels());
    // start googlenet and next fetch
    parallel_googlenet->run_async();
    fetch->run_async();
    fetch->sync();
    parallel_googlenet->sync();
    // verbose
    MPI_LOG( << "Iteration: " << iter);
    if (iter == 100) {
      start = std::chrono::system_clock::now();
    }
    if (iter == 200) {
      duration = std::chrono::duration_cast<std::chrono::milliseconds>
          (std::chrono::system_clock::now() - start);
      MPI_LOG( << "Batch Size: " << batch_size <<
          " Time Elapsed: " << duration.count());
    }
  }
  // delete
  fetch.reset();
  parallel_googlenet.reset();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // initilize MPI
  int ret;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
  time(40);
  time(48);
  time(56);
  // Finalize MPI
  MPI_CHECK(MPI_Finalize());
  return 0;
}
