//
// Created by dreame on 2025/7/29.
//
#ifndef PHOTOAPP_CVUTILS_H
#define PHOTOAPP_CVUTILS_H

#include <opencv2/opencv.hpp>
#include <android/bitmap.h>


class CvUtils {
public:
    static cv::Mat bitmapToMat(const AndroidBitmapInfo bitmapInfo, const void* pixels);

};

#endif //PHOTOAPP_CVUTILS_H
