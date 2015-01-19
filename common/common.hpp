// Copyright Lin Min 2014

#ifndef PURINE_COMMON
#define PURINE_COMMON

#ifndef DTYPE
#define DTYPE float
#endif

#include <vector>
#include <string>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <glog/logging.h>

using std::string;
using std::vector;

namespace purine {

/**
 * @macro UV_CHECK(condition) check the return code of libuv api function calls.
 *        Error is reported if execution is not successful.
 */
#define UV_CHECK(condition) \
  do { \
    int error = condition; \
    CHECK_EQ(error, 0) << " " << uv_strerror(error); \
  } while(0)

/**
 * @macro MPI_CHECK(condition) check the return code of mpi api function calls.
 *        Error is reported if execution is not successful.
 */
#define MPI_CHECK(condition) \
  do { \
    int error = condition; \
    CHECK_EQ(error, MPI_SUCCESS) << " " << mpi_strerror(error); \
  } while(0)

/**
 * @brief return the mpi error string from the error code.
 */
string mpi_strerror(int errorcode);

/**
 * @brief string get environment variable
 */
string get_env(const string& env);

/**
 * @fn int64_t cluster_seedgen()
 * @brief returns a random seed generated according to
 * /dev/random, pid, tid and time
 */
int64_t cluster_seedgen();

/**
 * @var typedef boost::mt19937 as purine::rng_t
 */
typedef boost::mt19937 rng_t;

/**
 * @fn rng_t* caffe_rng()
 * @brief returns cpu random number generator.
 */
rng_t* caffe_rng();

/**
 * @fn first_half
 * @brief take the first half of a vector
 */
template <typename T>
vector<T> first_half(const vector<T>& vec) {
  CHECK_EQ(vec.size() % 2, 0);
  int middle = vec.size() / 2;
  vector<T> ret;
  ret.insert(ret.end(), vec.begin(), vec.begin() + middle);
  return ret;
}

/**
 * @fn second_half
 * @brief take the second half of a vector
 */
template <typename T>
vector<T> second_half(const vector<T>& vec) {
  CHECK_EQ(vec.size() % 2, 0);
  int middle = vec.size() / 2;
  vector<T> ret;
  ret.insert(ret.end(), vec.begin() + middle, vec.end());
  return ret;
}

/**
 * @fn int current_rank()
 * @brief returns rank of current machine
 */
int current_rank();

}

#endif
