// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/googlenet.hpp"
#include "composite/graph/compute_loss.hpp"

int batch_size = 89;
string source = "/temp/imagenet-center-test-lmdb";
string mean_file = "/temp/imagenet-train-mean";

using namespace purine;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // initilize MPI
  int ret;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));
  // parallels
  // parameter server
  // fetch image
  shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
      false, false, true, batch_size, 224, vector<pair<int, int> >{{0, 0}});
  fetch->run();
  // create data parallelism of Googlenet;
  shared_ptr<ComputeLoss<GoogLeNet<true> > > googlenet_test
      = make_shared<ComputeLoss<GoogLeNet<true> > >(0, 0);
  // do the initialization
  googlenet_test->
      load("./googlenet_0.0001/googlenet_no_aux_dump_iter_100000.snapshot");
  // iteration
  for (int iter = 1; iter <= 542; ++iter) {
    // feed prefetched data to googlenet
    googlenet_test->feed(fetch->images(), fetch->labels());
    // start googlenet and next fetch
    googlenet_test->run_async();
    fetch->run_async();
    fetch->sync();
    googlenet_test->sync();
    // verbose
    googlenet_test->print_loss();
  }
  // delete
  fetch.reset();
  googlenet_test.reset();
  // Finalize MPI
  MPI_CHECK(MPI_Finalize());
  return 0;
}
