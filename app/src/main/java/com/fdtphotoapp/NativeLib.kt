package com.fdtphotoapp

import android.graphics.Bitmap

/**
 * @Author: zhanghongxin2
 * @Date: 2025/7/29 17:44
 * @Description:
 */
object NativeLib {
    init {
        System.loadLibrary("fdtphotoapp")
    }

    external fun loadBitmap(bitmap: Bitmap): Int

    external fun releaseBitmap(): Int
    external fun stringFromJNI(): String

}