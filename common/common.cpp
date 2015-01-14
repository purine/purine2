// Copyright Lin Min 2014

#include <stdint.h>
#include <fcntl.h>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <glog/logging.h>
#include <cstdlib>
#include <arpa/inet.h>
#include <sys/syscall.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include "common/common.hpp"

using std::fstream;
using std::ios;
using std::string;

namespace purine {

string get_env(const string& env) {
  const char* env_value = getenv(env.c_str());
  CHECK(env_value) << "Environment Variable " << env << " is not defined";
  return env_value;
}

int64_t cluster_seedgen(void) {
  int64_t s, seed, pid, tid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  tid = syscall(SYS_gettid);
  s = time(NULL);
  // TODO: how to hash with tid
  seed = abs(((s * 181) * ((pid + tid - 83) * 359)) % 104729);
  return seed;
}

rng_t* caffe_rng() {
  static thread_local rng_t rng_(cluster_seedgen());
  return &rng_;
}

}
