// Copyright Lin Min 2015
#include "dispatch/graph.hpp"

namespace purine {

static vector<vector<Blob*> > transpose(const vector<vector<Blob*> >& data) {
  vector<vector<Blob*> > result(data[0].size(), vector<Blob*>(data.size()));
  for (int i = 0; i < data[0].size(); i++) {
    for (int j = 0; j < data.size(); j++) {
      result[i][j] = data[j][i];
    }
  }
  return result;
}

template <typename T>
class Vectorize : public Graph {
  template <typename X, typename Y>
  friend Vectorize<Y>& operator >> (Vectorize<X>& v1, Vectorize<Y>& v2);
  template <typename X>
  friend Vectorize<X>& operator >> (const vector<vector<Blob*> >& bottom,
      Vectorize<X>& v);
  template <typename X>
  friend const vector<vector<Blob*> >& operator >> (Vectorize<X>& v,
      const vector<vector<Blob*> >& top);
 protected:
  vector<typename T::param_tuple> args_;
  vector<int> rank_;
  vector<int> device_;
  bool graphs_setup_ = false;
  bool transpose_;
  vector<T*> graphs_;
 public:
  Vectorize(int r, int d, const vector<int>& rank, const vector<int>& device,
      const vector<typename T::param_tuple>& args, bool transpose = true);
  Vectorize(int r, int d, const vector<int>& rank,
      const vector<int>& device, bool transpose = true);
  virtual ~Vectorize();
  void set_bottom(const vector<vector<Blob*> >& bottom);
  vector<vector<Blob*> > bottom();
  void set_top(const vector<vector<Blob*> >& top);
  vector<vector<Blob*> > top();

};

template <typename T>
Vectorize<T>::Vectorize(int r, int d,
    const vector<int>& rank, const vector<int>& device,
    const vector<typename T::param_tuple>& args, bool transpose)
    : transpose_(transpose) {
  CHECK_EQ(args.size(), rank.size());
  CHECK_EQ(args.size(), device.size());
  // create graph
  graphs_ = vector<T*>(args.size());
  for (int i = 0; i < args.size(); ++i) {
    graphs_[i] = createGraph<T>("...", rank[i], device[i], args[i]);
  }
}

template <typename T>
Vectorize<T>::Vectorize(int r, int d,
    const vector<int>& rank, const vector<int>& device,
    bool transpose) : rank_(rank), device_(device),
                             transpose_(transpose) {
  CHECK_EQ(device.size(), rank.size());
  graphs_ = vector<T*>(rank.size());
  for (int i = 0; i < rank.size(); ++i) {
    graphs_[i] = createGraph<T>(typeid(T).name(), rank[i], device[i]);
  }
}

template <typename T>
Vectorize<T>::~Vectorize() {
}

template <typename T>
void Vectorize<T>::set_bottom(const vector<vector<Blob*> >& bottom) {
  // check
  CHECK_GT(bottom.size(), 0);
  CHECK_GT(bottom[0].size(), 0);
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i].size(), bottom[0].size());
  }

  vector<vector<Blob*> > bottom_;
  if (this->transpose_) {
    bottom_ = transpose(bottom);
  }
  // create graph
  CHECK_EQ(bottom_.size(), graphs_.size());
  for (int i = 0; i < bottom_.size(); ++i) {
    bottom_[i] >> *graphs_[i];
  }
}

template <typename T>
vector<vector<Blob*> > Vectorize<T>::bottom() {
  vector<vector<Blob*> > tmp(graphs_.size());
  transform(graphs_.begin(), graphs_.end(), tmp.begin(),
      [](Connectable* g)->vector<Blob*>{
        return g->bottom();
      });
  if (this->transpose_) {
    return transpose(tmp);
  } else {
    return tmp;
  }
}

template <typename T>
void Vectorize<T>::set_top(const vector<vector<Blob*> >& top) {
  vector<vector<Blob*> > top_;
  if (this->transpose_) {
    top_ = transpose(top);
  }
  CHECK_EQ(top_.size(), graphs_.size());
  for (int i = 0; i < top_.size(); ++i) {
    *graphs_[i] >> top_[i];
  }
}

template <typename T>
vector<vector<Blob*> > Vectorize<T>::top() {
  vector<vector<Blob*> > tmp(graphs_.size());
  transform(graphs_.begin(), graphs_.end(), tmp.begin(),
      [](Connectable* g)->vector<Blob*>{
        return g->top();
      });
  if (this->transpose_) {
    tmp = transpose(tmp);
  }
  return tmp;
}

template <typename X>
Vectorize<X>& operator >> (const vector<vector<Blob*> >& bottom,
    Vectorize<X>& v) {
  v.set_bottom(bottom);
}

template <typename X>
const vector<vector<Blob*> >& operator >> (Vectorize<X>& v,
    const vector<vector<Blob*> >& top) {
  v.set_top(top);
}

template <typename X, typename Y>
Vectorize<Y>& operator >> (Vectorize<X>& v1, Vectorize<Y>& v2) {
  v1.top() >> v2;
}

}
