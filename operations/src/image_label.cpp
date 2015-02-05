// Copyright Lin Min 2015
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "operations/include/image_label.hpp"

using caffe::BlobProto;
using caffe::Datum;

namespace purine {

static void TensorFromBlob(const BlobProto& proto, Tensor* tensor) {
  Size s = tensor->size();
  DTYPE* data_vec = tensor->mutable_cpu_data();
  for (int i = 0; i < s.count(); ++i) {
    data_vec[i] = proto.data(i);
  }
}

ImageLabel::ImageLabel(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(source, mean, mirror, random, color, offset, interval,
      batch_size, crop_size)
      = args;
  CHECK_EQ(batch_size, outputs_[0]->size().num());
  CHECK_EQ(batch_size, outputs_[1]->size().num());
  CHECK_EQ(crop_size, outputs_[0]->size().height());
  CHECK_EQ(crop_size, outputs_[0]->size().width());
  mean_.reset(new Tensor(current_rank(), -1,
          {1, color ? 3 : 1, crop_size, crop_size}));
  if (!(mean == "")) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean, &blob_proto);
    TensorFromBlob(blob_proto, mean_.get());
  } else {
    caffe::caffe_memset(mean_->size().count() * sizeof(DTYPE), 0,
        mean_->mutable_cpu_data());
  }
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS)
      << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);
  CHECK_EQ(mdb_env_open(mdb_env_, source.c_str(), MDB_RDONLY|MDB_NOTLS,
          0664), MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_),
      MDB_SUCCESS) << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
      << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
      << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb " << source;
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
      MDB_SUCCESS) << "mdb_cursor_get failed";
  // go to the offset
  for (int i = 0; i < offset; ++i) {
    if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
        != MDB_SUCCESS) {
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
              MDB_FIRST), MDB_SUCCESS);
    }
  }
}

void ImageLabel::compute_cpu(const vector<bool>& add) {
  Datum datum;
  const DTYPE* mean = mean_->data();
  DTYPE* top_data = outputs_[0]->mutable_cpu_data();
  DTYPE* top_label = outputs_[1]->mutable_cpu_data();
  int size = outputs_[0]->size().count() / outputs_[0]->size().num();
  const int mean_height = mean_->size().height();
  const int mean_width = mean_->size().width();
  const int mean_channels = mean_->size().channels();
  // mean h_off w_off
  const int mean_h_off = (mean_height - crop_size) / 2;
  const int mean_w_off = (mean_width - crop_size) / 2;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
            MDB_GET_CURRENT), MDB_SUCCESS);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    const string& data = datum.data();
    int height = datum.height();
    int width = datum.width();
    int channels = datum.channels();

    if (data.size()) {
      int h_off, w_off;
      if (random) {
        h_off = caffe::caffe_rng_rand() % (height - crop_size);
        w_off = caffe::caffe_rng_rand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }

      if (mirror && caffe::caffe_rng_rand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                  * crop_size + (crop_size - 1 - w);
              int data_index = (c * height + h + h_off) * width + w + w_off;
              int mean_index = (c * mean_height + h + mean_h_off) * mean_width
                  + w + mean_w_off;
              DTYPE datum_element =
                  static_cast<DTYPE>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = datum_element - mean[mean_index];
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                  * crop_size + w;
              int data_index = (c * height + h + h_off) * width + w + w_off;
              int mean_index = (c * mean_height + h + mean_h_off) * mean_width
                  + w + mean_w_off;
              DTYPE datum_element =
                  static_cast<DTYPE>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = datum_element - mean[mean_index];
            }
          }
        }
      }
    } else {
      for (int j = 0; j < size; ++j) {
        top_data[item_id * size + j] = datum.float_data(j) - mean[j];
      }
    }
    top_label[item_id] = datum.label();
    if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
        != MDB_SUCCESS) {
      // We have reached the end. Restart from the first.
      // DLOG(INFO) << "Restarting data prefetching from start.";
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
              MDB_FIRST), MDB_SUCCESS);
    }
  }

  for (int i = 0; i < interval - batch_size; ++i) {
    if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
        != MDB_SUCCESS) {
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
              MDB_FIRST), MDB_SUCCESS);
    }
  }
}

ImageLabel::~ImageLabel() {
  mdb_cursor_close(mdb_cursor_);
  mdb_close(mdb_env_, mdb_dbi_);
  mdb_txn_abort(mdb_txn_);
  mdb_env_close(mdb_env_);
}

}
