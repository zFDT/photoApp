# photoApp

Android 照片应用 Demo：演示 TensorFlow Lite 通过 JNI（C++）集成到 Android 工程的最小示例。

## 功能

- `NativeLib`：JNI 桥接，`stringFromJNI()` 示例
- TensorFlow Lite C++ 依赖集成（`tensorflow-lite` / `tensorflow-lite-support` / `tensorflow-lite-metadata`）
- Python 侧模型转换脚本（`pth2tflite.py`，PyTorch → TFLite）

## 目录

```
app/src/main/java/com/fdtphotoapp/   # Android 端（MainActivity / NativeLib）
python/                               # Python 模型转换与依赖安装
```

## TensorFlow Lite C++ 获取方式（备忘）

1. 通过 Android Studio 的 `libs.versions.toml` 引入：
   ```toml
   tensorflow-lite = { group = "org.tensorflow", name = "tensorflow-lite", version = "2.16.0" }
   tensorflow-lite-support = { group = "org.tensorflow", name = "tensorflow-lite-support", version = "0.4.4" }
   tensorflow-lite-metadata = { group = "org.tensorflow", name = "tensorflow-lite-metadata", version = "0.4.4" }
   ```
2. 详见 `remared.md`（本仓库历史备忘）

> 说明：`remared.md` 为早期误命名的 `readme.md`，保留作历史参考。
