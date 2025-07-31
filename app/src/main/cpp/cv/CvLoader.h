//
// Created by dreame on 2025/7/29.
//

#ifndef PHOTOAPP_CVLOADER_H
#define PHOTOAPP_CVLOADER_H

#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include "CvUtils.h"

class CvLoader {
public:
    int storeBitmap(JNIEnv *env, jobject bitmap);

    int releaseStoredBitmap();

private:
    AndroidBitmapInfo bitmapInfo; // Bitmap 信息
    void *pixels = nullptr;         // Bitmap 像素数据指针
    cv::Mat originalMat; // 存储原始图像数据
};

#endif //PHOTOAPP_CVLOADER_H
