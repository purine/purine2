// Copyright Lin Min 2015
#ifndef PURINE_FETCH_IMAGE
#define PURINE_FETCH_IMAGE

#include <vector>
#include <utility>
#include "dispatch/runnable.hpp"

using namespace std;

namespace purine {

class FetchImage : public Runnable {
 protected:
  vector<Blob*> images_;
  vector<Blob*> labels_;
 public:
  FetchImage(const string& source, const string& mean,
      bool mirror, bool random, bool color, int batch_size, int crop_size,
      const vector<pair<int, int> >& location);
  virtual ~FetchImage() override {}
  const vector<Blob*>& images() { return images_; }
  const vector<Blob*>& labels() { return labels_; }
};

}

#endif
