//
// Created by dreame on 2025/7/29.
//
#include <string.h>
#include "CvLoader.h"
#include "android/log.h"
#include "malloc.h"

// 用宏定义来打印日志
#define LOG_TAG "CvLoader"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)


int CvLoader::storeBitmap(JNIEnv *env, jobject bitmap) {
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
        LOGE("获取Bitmap信息错误!");
        return 1;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        LOGE("锁定Bitmap像素数据失败!");
        return 1;
    }
    size_t bufferSize = bitmapInfo.height * bitmapInfo.stride;
    // void* 是一个通用指针类型，可以指向任何类型的数据；不包含类型信息，需要格外小心使用。
    void *pixelBuffer = malloc(bufferSize);
    if (pixelBuffer != nullptr) {
        memcpy(pixelBuffer, pixels, bufferSize);
        pixels = pixelBuffer; // 将像素数据指针指向新分配的内存
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    LOGI("Bitmap存储成功，宽度: %d, 高度: %d, stride: %d", bitmapInfo.width, bitmapInfo.height, bitmapInfo.stride);
    originalMat = CvUtils::bitmapToMat(bitmapInfo, pixels);
    if (originalMat.empty()) {
        LOGE("Bitmap转Mat时出现错误!");
        return 1;
    } else {
        LOGI("Bitmap转Mat成功");
        return 0;
    }
}

int CvLoader::releaseStoredBitmap() {
    if (pixels != nullptr) {
        free(pixels); // 释放之前分配的内存
        pixels = nullptr; // 重置指针
        LOGI("Bitmap像素数据已释放");
        return 0;
    } else {
        LOGE("没有可释放的Bitmap像素数据");
        return 1;
    }
}