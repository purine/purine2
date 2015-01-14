#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

#include "caffeine/caffeine.hpp"
#include "common/common.hpp"

namespace caffe {

void im2col_cpu(const DTYPE* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, DTYPE* data_col);

void col2im_cpu(const DTYPE* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, DTYPE* data_im, bool add);

void im2col_gpu(const DTYPE* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, DTYPE* data_col);

void col2im_gpu(const DTYPE* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, DTYPE* data_im, bool add);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
