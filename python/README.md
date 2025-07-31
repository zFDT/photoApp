# PyTorch到TensorFlow Lite转换工具

## 功能说明
将PyTorch模型(.pth)转换为TensorFlow Lite模型(.tflite)，用于移动端部署。

## 安装依赖

### 方法1：自动安装（推荐）
```bash
python install_dependencies.py
```

### 方法2：使用requirements.txt
```bash
pip install -r requirements.txt
```

### 方法3：手动安装
```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch (根据你的系统选择合适版本)
pip install torch torchvision

# 安装ONNX相关
pip install onnx onnx-tf

# 安装TensorFlow
pip install tensorflow

# 安装其他依赖
pip install numpy protobuf
```

## 常见问题

### 1. onnx-tf安装失败
```bash
# 尝试指定版本
pip install onnx-tf==1.9.0

# 或使用conda
conda install -c conda-forge onnx-tf
```

### 2. TensorFlow GPU版本
如果需要GPU加速：
```bash
pip install tensorflow-gpu
```

### 3. PyTorch CPU版本
如果只需要CPU版本：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. protobuf版本冲突
```bash
pip install protobuf==3.20.3
```

## 使用方法

### 基本用法
```python
python pth2tflite.py
```

### 自定义参数
修改 `pth2tflite.py` 中的配置：
```python
model_path = 'your_model.pth'
input_shape = (1, 3, 224, 224)  # 根据模型调整
output_name = 'your_model.tflite'
```

## 支持的模型格式
- 完整PyTorch模型 (.pth)
- PyTorch state_dict (需要模型架构)
- TorchScript模型

## 输出文件
- `model.tflite`: 最终的TensorFlow Lite模型
- 临时文件会自动清理

## 注意事项
1. 确保输入维度与原模型匹配
2. 某些PyTorch操作可能不支持转换到TFLite
3. 转换过程可能需要较长时间，取决于模型大小
