// Copyright Lin Min 2015
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

  int interval = batch_size * location.size();

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
  int offset = 0;
  for (auto kv : images) {
    int size = batch_size * kv.second.size();
    Blob* image = create("IMAGES", kv.first, -1,
        {size, color ? 3 : 1, crop_size, crop_size});
    Blob* label = create("LABELS", kv.first, -1, {size, 1, 1, 1});
    Op<ImageLabel>* image_label = create<ImageLabel>("FETCH", kv.first, -1,
        "fetch", ImageLabel::param_tuple(source, mean, mirror, random, color,
            interval, offset, size, crop_size));
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
