// Copyright Lin Min 2015
#include "dispatch/blob.hpp"
#include "dispatch/graph_template.hpp"
#include "operations/include/mem_copy.hpp"
#include "operations/include/eltwise.hpp"
#include "dispatch/op_template.hpp"
#include "composite/graph/copy.hpp"
#include "composite/vectorize.hpp"

namespace purine {
typedef vector<Blob*> B;
void Copy::setup() {
  static int tag = 0;
  CHECK(bottom_setup_);

  // check top
  if (top_.size() != 0) {
    CHECK_EQ(bottom_.size(), 1);
    CHECK_EQ(top_.size(), 1);
  } else {
    CHECK_EQ(bottom_.size(), 1);
    top_ = vector<Blob*>(bottom_.size());
    if (rank_ == bottom_[0]->rank() &&
        device_ == bottom_[0]->device()) {
      top_[0] = bottom_[0];
    } else {
      top_[0] = create(bottom_[0]->tensor()->size(), "dest", rank_, device_);
    }
  }
  // setup internal routes
  if (bottom_[0] == top_[0]) {
    return;
  }
  if (bottom_[0]->rank() == top_[0]->rank()) { // on the same machine
    B{ bottom_[0] } >> *create<MemCopy>(MemCopy::param_tuple(),
        "memcopy", "main") >> B{ top_[0] };
  } else { // on different machines
    Blob* src;
    if (bottom_[0]->device() < 0) {
      src = bottom_[0];
    } else {
      src = create(bottom_[0]->tensor()->size(), "src_cpu",
          bottom_[0]->rank(), -1);
      B{ bottom_[0] } >> *create<MemCopy>(MemCopy::param_tuple(),
          "memcopy", "outbound") >> B{ src };
    }
    Blob* dest;
    if (top_[0]->device() < 0) {
      dest = top_[0];
    } else {
      dest = create(top_[0]->tensor()->size(), "dest_cpu",
          top_[0]->rank(), -1);
      B{ dest } >> *create<MemCopy>(MemCopy::param_tuple(),
          "memcopy", "inbound") >> B{ top_[0] };
    }
    B{ src } >> *create<Isend>(Isend::param_tuple(tag, dest->rank()),
        "isend", src->rank(), src->device(), "main");
    *create<Irecv>(Irecv::param_tuple(tag++, src->rank()),
        "irecv", dest->rank(), dest->device(), "main") >> B{ dest };
  }
}

Connectable& operator >> (Copy& copy, Connectable& graph) {
  int num = copy.bottom().size();
  copy.rank_ = graph.rank();
  copy.device_ = graph.device();
  return copy.top() >> graph;
}

Copy& operator >> (Connectable& graph, Copy& copy) {
  copy.set_bottom(graph.top());
  return copy;
}

Copy& operator >> (Copy& copy1, Copy& copy2) {
  LOG(FATAL) << "Connecting Copy graphs";
}

Copy& operator >> (const vector<Blob*>& inputs, Copy& copy) {
  copy.set_bottom(inputs);
  return copy;
}

const vector<Blob*>& operator >> (Copy& copy, const vector<Blob*>& outputs) {
  copy.set_top(outputs);
  return outputs;
}

void Distribute::setup() {
  CHECK(bottom_setup_);
  // check top
  if (top_.size() != 0) {
    for (Blob* t : top_) {
      CHECK_EQ(t->tensor()->size(), bottom_[0]->tensor()->size());
    }
  } else {
    top_ = vector<Blob*>(rank_device.size());
    for (int i = 0; i < rank_device.size(); ++i) {
      if (rank_device[i].first == bottom_[0]->rank() &&
          rank_device[i].second == bottom_[0]->device()) {
        top_[i] = bottom_[0];
      } else {
        top_[i] = create(bottom_[0]->tensor()->size(), "...",
            rank_device[i].first, rank_device[i].second);
      }
    }
  }
}

void Aggregate::setup() {
  CHECK(bottom_setup_);
  // check top
  if (top_.size() != 0) {
    CHECK_EQ(top_.size(), 1);
    for (Blob* b : bottom_) {
      CHECK_EQ(b->tensor()->size(), top_[0]->tensor()->size());
    }
  } else {
    top_ = {
      create(bottom_[0]->tensor()->size(), "top", rank_, device_)
    };
  }

  map<int, Aggregate*> local_aggs;
  map<int, vector<Blob*> > local_blobs;
  vector<Blob*> dest_blobs;
  for (int i = 0; i < bottom_.size(); ++i) {
    if (bottom_[i]->rank() != top_[0]->rank() &&
        local_aggs.count(bottom_[i]->rank()) == 0) {
      local_aggs[bottom_[i]->rank()] = createGraph<Aggregate>("local_agg",
          bottom_[i]->rank(), -1, Type::SUM);
      local_blobs[bottom_[i]->rank()] = { bottom_[i] };
    } else if (bottom_[i]->rank() != top_[0]->rank()) {
      local_blobs[bottom_[i]->rank()].push_back(bottom_[i]);
    } else {
      dest_blobs.push_back(bottom_[i]);
    }
  }

  for (auto kv : local_aggs) {
    local_blobs[kv.first] >> *(kv.second);
  }
  vector<Blob*> agged(local_aggs.size());
  transform(local_aggs.begin(), local_aggs.end(), agged.begin(),
      [](std::pair<const int, Aggregate*>& agg)->Blob* {
        return agg.second->top()[0];
      });
  dest_blobs.insert(dest_blobs.end(), agged.begin(), agged.end());

  auto copy = createGraph<Vectorize<Copy> >("...",
      vector<int>(dest_blobs.size(), top_[0]->rank()),
      vector<int>(dest_blobs.size(), top_[0]->device()), true);
  vector<vector<Blob*> >{ dest_blobs } >> *copy;
  vector<Blob*> copied = copy->top()[0];

  copied >> *create<Sum>(Sum::param_tuple(), "sum", top_[0]->rank(),
      top_[0]->device(), "main") >> top_;
}

}
