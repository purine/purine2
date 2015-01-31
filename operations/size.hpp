// Copyright Lin Min 2014

#ifndef PURINE_SIZE
#define PURINE_SIZE

#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <initializer_list>

using std::ostream;
using std::vector;
using std::initializer_list;

namespace purine {

class Size {
 private:
  int num_;
  int channels_;
  int height_;
  int width_;
 public:
  explicit Size() {
    num_ = 0;
    channels_ = 0;
    height_ = 0;
    width_ = 0;
  }
  explicit Size(int num, int channels, int height, int width) {
    num_ = num;
    channels_ = channels;
    height_ = height;
    width_ = width;
  }
  Size(const vector<int>& size) {
    CHECK_EQ(size.size(), 4);
    num_ = size[0];
    channels_ = size[1];
    height_ = size[2];
    width_ = size[3];
  }
  Size(const Size &size) {
    num_ = size.num();
    channels_ = size.channels();
    height_ = size.height();
    width_ = size.width();
  }
  Size(const initializer_list<int> list)
      : Size(vector<int>(list)) {
  }

  inline int num() const { return num_; }
  inline int channels() const { return channels_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int count() const { return width_ * height_ * channels_ * num_; }

  /**
   * @brief the default constructed size is invalid: all dimensions are zero
   */
  inline bool is_valid() const {
    return (num_ != 0 && channels_ != 0 && height_ != 0 && width_ != 0);
  }

  inline bool operator == (const Size& other) const {
    return (other.num() == num_ && other.channels() == channels_
        && other.width() == width_ && other.height() == height_);
  }
  inline bool operator < (const Size& other) const {
    return count() < other.count();
  }
  inline bool operator > (const Size& other) const {
    return !(operator<(other) || operator==(other));
  }
  inline bool operator <= (const Size& other) const {
    return operator<(other) || operator==(other);
  }
  inline bool operator >= (const Size& other) const {
    return operator>(other) || operator==(other);
  }
  friend ostream& operator<<(ostream& os, const Size& s);
  friend Size operator + (Size size, const Size& add);
  inline Size& operator += (const Size& add) {
    num_ += add.num_;
    channels_ += add.channels_;
    height_ += add.height_;
    width_ += add.width_;
    return *this;
  }
};

/**
 * @brief print out the size in the format [num, channels, height, width]
 */
inline ostream& operator<<(ostream& os, const Size& s) {
  os << "(" << s.num_ << ',' << s.channels_ << ','
     << s.height_ << "," << s.width_ << ")";
  return os;
}

inline Size operator + (Size size, const Size& add) {
  size.num_ += add.num_;
  size.channels_ += add.channels_;
  size.height_ += add.height_;
  size.width_ += add.width_;
  return size;
}

class Stride {
 public:
  explicit Stride() {
    wstride_ = 0;
    hstride_ = 0;
    cstride_ = 0;
    nstride_ = 0;
  }
  explicit Stride(const Size& size) {
    wstride_ = 1;
    hstride_ = size.width();
    cstride_ = size.width() * size.height();
    nstride_ = size.width() * size.height() * size.channels();
  }
  explicit Stride(const int n, const int c, const int h, const int w) {
    nstride_ = n;
    cstride_ = c;
    hstride_ = h;
    wstride_ = w;
  }
  explicit Stride(const vector<int>& stride) {
    CHECK_EQ(stride.size(), 4);
    nstride_ = stride[0];
    cstride_ = stride[1];
    hstride_ = stride[2];
    wstride_ = stride[3];
  }
  Stride(const initializer_list<int> list)
      : Stride(vector<int>(list)) {
  }
  inline int nstride() const { return nstride_; }
  inline int cstride() const { return cstride_; }
  inline int hstride() const { return hstride_; }
  inline int wstride() const { return wstride_; }
  inline bool operator == (const Stride& other) const {
    return (other.nstride_ == nstride_ && other.cstride_ == cstride_
        && other.wstride_ == wstride_ && other.hstride_ == hstride_);
  }
  friend ostream& operator<<(ostream& os, const Stride& s);
 protected:
  int nstride_;
  int cstride_;
  int hstride_;
  int wstride_;
};

inline ostream& operator<<(ostream& os, const Stride& s) {
  os << "(" << s.nstride_ << ',' << s.cstride_ << ','
     << s.hstride_ << "," << s.wstride_ << ")";
  return os;
}

class Offset {
 public:
  explicit Offset(const int n = 0, const int c = 0,
      const int h = 0, const int w = 0) {
    noffset_ = n;
    coffset_ = c;
    hoffset_ = h;
    woffset_ = w;
  }
  explicit Offset(const vector<int>& offset) {
    CHECK_EQ(offset.size(), 4);
    noffset_ = offset[0];
    coffset_ = offset[1];
    hoffset_ = offset[2];
    woffset_ = offset[3];
  }
  Offset(const initializer_list<int> list)
      : Offset(vector<int>(list)) {
  }
  inline int noffset() const { return noffset_; }
  inline int coffset() const { return coffset_; }
  inline int hoffset() const { return hoffset_; }
  inline int woffset() const { return woffset_; }
  inline bool operator == (const Offset& other) const {
    return (other.noffset_ == noffset_ && other.coffset_ == coffset_
        && other.woffset_ == woffset_ && other.hoffset_ == hoffset_);
  }
  friend ostream& operator<<(ostream& os, const Offset& s);
  friend Offset operator+ (Offset offset, const Offset& add);
  inline Offset& operator+= (const Offset& add) {
    noffset_ += add.noffset_;
    coffset_ += add.coffset_;
    hoffset_ += add.hoffset_;
    woffset_ += add.woffset_;
    return *this;
  }
 protected:
  int noffset_;
  int coffset_;
  int hoffset_;
  int woffset_;
};

inline ostream& operator<<(ostream& os, const Offset& s) {
  os << "(" << s.noffset_ << ',' << s.coffset_ << ','
     << s.hoffset_ << "," << s.woffset_ << ")";
  return os;
}

inline Offset operator+ (Offset offset, const Offset& add) {
  offset.noffset_ += add.noffset_;
  offset.coffset_ += add.coffset_;
  offset.hoffset_ += add.hoffset_;
  offset.woffset_ += add.woffset_;
  return offset;
}

}

#endif
