// Copyright Lin Min 2014

#define CATCH_CONFIG_RUNNER
#include <mpi.h>
#include "catch/catch.hpp"

int main(int argc, char** argv) {
  int ret;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
