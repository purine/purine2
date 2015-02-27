// Copyright Lin Min 2015
#include <lmdb.h>

#include "dispatch/op.hpp"
#include "operations/include/image_label.hpp"
#include "composite/graph/fetch_image.hpp"
#include "composite/graph/split.hpp"
#include "composite/graph/copy.hpp"
#include "composite/vectorize.hpp"

namespace purine {

FetchImage::FetchImage(const string& source, const string& mean, bool mirror,
    bool random, bool color, int batch_size, int crop_size,
    const vector<pair<int, int> >& location) {
  map<int, vector<Blob*> > images;
  map<int, vector<Blob*> > labels;

  for (const pair<int, int>& loc : location) {
    Blob* image = create("image", loc.first, loc.second,
        {batch_size, color ? 3 : 1, crop_size, crop_size});
    Blob* label = create("label", loc.first, loc.second, {batch_size, 1, 1, 1});
    if (images.count(loc.first) != 0) {
      images[loc.first].push_back(image);
      labels[loc.first].push_back(label);
    } else {
      images[loc.first] = { image };
      labels[loc.first] = { label };
    }
    images_.push_back(image);
    labels_.push_back(label);
  }
  MDB_env* mdb_env_;
  MDB_stat mdb_stat_;
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS)
      << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);
  CHECK_EQ(mdb_env_open(mdb_env_, source.c_str(), MDB_RDONLY|MDB_NOTLS,
          0664), MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_env_stat(mdb_env_, &mdb_stat_), MDB_SUCCESS);
  int entries = mdb_stat_.ms_entries;
  mdb_env_close(mdb_env_);
  MPI_LOG( << "Lmdb contains " << entries << " entries." );
  // int each_location = entries / location.size();
  int offset = 0;
  for (auto kv : images) {

    MPI_LOG( << " ============================= " );
    MPI_LOG( << " machine " << kv.first );
    MPI_LOG( << " ============================= " );
    int size = batch_size * kv.second.size();
    int interval = batch_size * location.size();
    MPI_LOG( << " offset            " << offset );
    MPI_LOG( << " batch size        " << size );
    MPI_LOG( << " interval          " << interval );

    Blob* image = create("IMAGES", kv.first, -1,
        {size, color ? 3 : 1, crop_size, crop_size});
    Blob* label = create("LABELS", kv.first, -1, {size, 1, 1, 1});
    Op<ImageLabel>* image_label = create<ImageLabel>("FETCH", kv.first, -1,
        "fetch", ImageLabel::param_tuple(source, mean, mirror, random, color,
            offset, interval, size, crop_size));
    *image_label >> vector<Blob*>{ image, label };

    Split* split_image = createGraph<Split>("split_image", kv.first, -1,
        Split::param_tuple(Split::NUM),
        vector<int>(kv.second.size(), batch_size));
    Split* split_label = createGraph<Split>("split_label", kv.first, -1,
        Split::param_tuple(Split::NUM),
        vector<int>(kv.second.size(), batch_size));
    vector<Blob*>{ image } >> *split_image;
    vector<Blob*>{ label } >> *split_label;

    // copy splitted images to the destination
    vector<vector<Blob*> >{ split_image->top() }
    >> *createAny<Vectorize<Copy> >("copy_image_to_dest",
        vector<Copy::param_tuple>(kv.second.size()))
           >> vector<vector<Blob*> >{ kv.second };

    vector<vector<Blob*> >{ split_label->top() }
    >> *createAny<Vectorize<Copy> >("copy_label_to_dest",
        vector<Copy::param_tuple>(kv.second.size()))
           >> vector<vector<Blob*> >{ labels[kv.first] };

    offset += size;
  }
}

}
