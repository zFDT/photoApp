#include <jni.h>
#include <string>
#include "cv/CvLoader.h"

CvLoader *cvLoader;

extern "C" JNIEXPORT jstring JNICALL
Java_com_fdtphotoapp_NativeLib_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
extern "C" JNIEXPORT jint JNICALL
Java_com_fdtphotoapp_NativeLib_loadBitmap(JNIEnv *env, jobject thiz, jobject bitmap) {
    if (cvLoader == nullptr) {
        cvLoader = new CvLoader();
    }
    return cvLoader->storeBitmap(env, bitmap);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_fdtphotoapp_NativeLib_releaseBitmap(JNIEnv *env, jobject thiz) {
    if (cvLoader != nullptr) {
        int result = cvLoader->releaseStoredBitmap();
        delete cvLoader; // 释放 CvLoader 实例
        cvLoader = nullptr; // 重置指针
        return result;
    }
    return 0; // 如果 cvLoader 为 nullptr，返回错误代码

}
