# photoApp — TensorFlow Lite + JNI 集成示例

Android 照片应用 Demo：演示 TensorFlow Lite 通过 JNI（C++）集成到 Android 工程的最小示例，打通「PyTorch → TFLite → C++ JNI → Android 推理」的完整链路。

## 项目简介

这是一个 **Android + TensorFlow Lite JNI（C++）集成最小示例**，用于验证在 Android 原生工程中通过 JNI 调用 C++ 侧 TensorFlow Lite 库、并在 Python 侧完成 PyTorch 模型到 TFLite 转换的完整技术链路。

链路如下：

```
PyTorch 模型 ──(pth2tflite.py)──▶ TFLite 模型 ──(JNI)──▶ C++ 推理 ──▶ Android 应用
```

## 核心功能 / 技术特性

### JNI 桥接
- `NativeLib`：通过 `stringFromJNI()` 示例演示 Java 与 C++ 之间的桥接调用，作为后续接入 TFLite 推理的入口。

### TensorFlow Lite 集成
- 引入 TensorFlow Lite 相关依赖（`tensorflow-lite` / `tensorflow-lite-support` / `tensorflow-lite-metadata`，核心版本 2.16.0）。
- 支持在 C++ 侧加载并执行 TFLite 模型推理。

### OpenCV Mobile
- C++ 层集成 OpenCV（`cv/CvLoader.cpp`、`cv/CvUtils.cpp`），通过 `lib/opencv-mobile-4.10.0-android` 提供图像预处理能力。

### 模型转换（pth2tflite）
- `python/pth2tflite.py`：完成 PyTorch → TFLite 的模型转换脚本，为移动端部署准备轻量化模型。

## 目录

```
app/src/main/cpp/                          # C++ 层
├── cv/CvLoader.cpp / cv/CvUtils.cpp       # OpenCV 图像处理
└── lib/opencv-mobile-4.10.0-android       # OpenCV Mobile 4.10.0
app/src/main/java/com/fdtphotoapp/         # Android 端（MainActivity / NativeLib）
python/                                    # Python 模型转换与依赖安装（pth2tflite.py）
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

## 技术特色与复盘

### 模型转换全链路
- 在 Python 侧用 `pth2tflite.py` 完成 PyTorch 到 TFLite 的转换，打通了「训练框架 → 移动端推理格式」的完整转换路径。
- 通过 `tensorflow-lite-support` 与 `tensorflow-lite-metadata` 引入模型元数据与支持库，为后续在端侧做预处理/后处理铺路。

### C++ JNI 集成
- 采用 JNI 桥接方式，让 Android 端能直接调用 C++ 侧的高性能推理逻辑，兼顾了移动端的性能诉求与原生工程的可控性。
- 通过 OpenCV Mobile 在 C++ 层完成图像处理，避免重复造轮子，也为模型推理前的高效预处理提供了基础。

### 为移动端 AI 打基础
- 项目虽为最小示例，但已覆盖「模型转换 + 原生推理 + 图像处理」三大核心环节，可作为后续接入真实模型（分类 / 检测 / 分割）的骨架。
- 目录与依赖结构清晰，便于在此基础上扩展多模型加载、量化优化、多线程推理等能力。

### 个人贡献
- 独立搭建 Android 原生工程并完成 JNI 桥接与 TFLite C++ 依赖集成。
- 编写 PyTorch → TFLite 转换脚本，验证并跑通模型转换全流程。
- 整理 TFLite 依赖获取方式与仓库备忘，形成可复用的工程模板。

### 收获与规划
- **收获**：深入理解了 JNI 跨语言调用机制、TFLite C++ 推理流程，以及移动端模型部署的完整链路。
- **规划**：后续计划接入真实检测/分类模型，补充模型量化、输入预处理与推理结果后处理，构建一个完整的端侧 AI 照片应用。
