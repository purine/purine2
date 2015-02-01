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
    Blob* image = create({batch_size, color ? 3 : 1, crop_size, crop_size},
        "image", loc.first, loc.second);
    Blob* label = create({batch_size, 1, 1, 1}, "label", loc.first, loc.second);
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
    Blob* image = create({size, color ? 3 : 1, crop_size, crop_size},
        "...", kv.first, -1);
    Blob* label = create({size, 1, 1, 1}, "...", kv.first, -1);
    Op<ImageLabel>* image_label = create<ImageLabel>(
        ImageLabel::param_tuple(source, mean, mirror, random, color,
            interval, offset, size, crop_size), "...", kv.first, -1, "fetch");
    *image_label >> vector<Blob*>{ image, label };

    Split* split_image = createGraph<Split>("...", kv.first, -1, Split::NUM,
        vector<int>(kv.second.size(), batch_size));
    Split* split_label = createGraph<Split>("...", kv.first, -1, Split::NUM,
        vector<int>(kv.second.size(), batch_size));
    vector<Blob*>{ image } >> *split_image;
    vector<Blob*>{ label } >> *split_label;

    // copy splitted images to the destination
    vector<vector<Blob*> >{ split_image->top() } >>
        *createGraph<Vectorize<Copy> >("...",
            vector<int>(kv.second.size(), 0),
            vector<int>(kv.second.size(), 0))
            >> vector<vector<Blob*> >{ kv.second };

    vector<vector<Blob*> >{ split_label->top() } >>
        *createGraph<Vectorize<Copy> >("...",
            vector<int>(kv.second.size(), 0),
            vector<int>(kv.second.size(), 0))
            >> vector<vector<Blob*> >{ labels[kv.first] };
    offset += size;
  }
}

}
