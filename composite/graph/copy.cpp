// Copyright Lin Min 2015
#include "dispatch/blob.hpp"
#include "operations/include/mem_copy.hpp"
#include "dispatch/op_template.hpp"
#include "composite/graph/copy.hpp"

namespace purine {
typedef vector<Blob*> B;
void Copy::setup() {
  CHECK(bottom_setup_);

  // check top
  if (top_.size() != 0) {
    CHECK_EQ(bottom_.size(), top_.size());
  } else {
    CHECK_EQ(ranks_.size(), bottom_.size());
    CHECK_EQ(devices_.size(), bottom_.size());
    top_ = vector<Blob*>(bottom_.size());
    for (int i = 0; i < bottom_.size(); ++i) {
      if (ranks_[i] == bottom_[i]->rank() &&
          devices_[i] == bottom_[i]->device()) {
        top_[i] = bottom_[i];
      } else {
        top_[i] = create(bottom_[i]->tensor()->size(), "dest",
            ranks_[i], devices_[i]);
      }
    }
  }
  for (int i = 0; i < bottom_.size(); ++i) {
    if (bottom_[i]->rank() == top_[i]->rank()) { // on the same machine
      B{ bottom_[i] } >> *create<MemCopy>(MemCopy::param_tuple(),
          "memcopy", "main") >> B{ top_[i] };
    } else { // on different machines
      Blob* src;
      if (bottom_[i]->device() < 0) {
        src = bottom_[i];
      } else {
        src = create(bottom_[i]->tensor()->size(), "src_cpu",
            bottom_[i]->rank(), -1);
        B{ bottom_[i] } >> *create<MemCopy>(MemCopy::param_tuple(),
            "memcopy", "outbound") >> B{ src };
      }
      Blob* dest;
      if (top_[i]->device() < 0) {
        dest = top_[i];
      } else {
        dest = create(top_[i]->tensor()->size(), "dest_cpu",
            top_[i]->rank(), -1);
        B{ dest } >> *create<MemCopy>(MemCopy::param_tuple(),
            "memcopy", "inbound") >> B{ top_[i] };
      }
      B{ src } >> *create<Isend>(Isend::param_tuple(), "isend", "main");
      *create<Irecv>(Irecv::param_tuple(), "irecv", "main") >> B{ dest };
    }
  }
}

CompositeGraph& operator >> (Copy& copy, CompositeGraph& graph) {
  int num = copy.bottom().size();
  copy.set_ranks(vector<int>(graph.rank(), num));
  copy.set_devices(vector<int>(graph.device(), num));
  return copy.top() >> graph;
}

void Distribute::setup() {
  CHECK(bottom_setup_);
  // check top
  if (top_.size() != 0) {
    LOG(FATAL) << " Do not give Composite Graph outputs";
  } else {
    top_ = vector<Blob*>(rank_device.size() * bottom_.size(), NULL);
    for (int j = 0; j < bottom_.size(); ++j) {
      Blob* b = bottom_[j];
      Size b_size = b->tensor()->size();
      map<int, Blob*> rank_cpu;
      for (int i = 0; i < rank_device.size(); ++i) {
        int rank = rank_device[i].first;
        int device = rank_device[i].second;
        Blob* dest;
        if (b->rank() == rank) {
          dest = create(b_size, "dest", rank, device);
          B{ b } >> *create<Copy>(Copy::param_tuple(), "copy") >> B{ dest };
        } else {
          if (rank_cpu.count(rank) == 0) {
            Blob* middle = create(b_size, "middle", rank, -1);
            rank_cpu[rank] = middle;
            B{ b } >> *create<Copy>(Copy::param_tuple(), "copy") >> B{ middle };
          }
          dest = create(b_size, "dest", rank, device);
          B{ rank_cpu[rank] } >> *create<Copy>(Copy::param_tuple(),
              "copy") >> B{ dest };
        }
        top_[j + i * bottom_.size()] = dest;
      }
    }
  }
}

vector<Blob*> Distribute::top(int index) {
  int count = bottom_.size();
  CHECK(bottom_setup_ && top_setup_);
  vector<Blob*> ret;
  ret.insert(ret.end(), top_.begin() + index * count,
      top_.begin() + (index + 1) * count);
  return ret;
}

}
