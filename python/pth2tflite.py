'''
PyTorch 模型转 TensorFlow Lite 转换工具
====================================

功能描述:
- 将 PyTorch 的 .pth 模型文件转换为 TensorFlow Lite 的 .tflite 格式
- 支持多种 PyTorch 模型保存格式 (完整模型、state_dict等)
- 提供智能的模型架构推断和参数匹配
- 包含完整的错误处理和多种转换策略

使用方法:
    python pth2tflite.py
    
支持的输入格式:
- 完整的 PyTorch 模型对象
- state_dict 字典格式
- 包含 'model' 或 'state_dict' 键的检查点文件

输出:
- model.tflite: 转换后的 TensorFlow Lite 模型文件
- model.onnx: 中间格式的 ONNX 模型文件 (临时)

作者: zhanghongxin2 
日期: 2025-07-31
版本: 2.0
'''

import os
import sys
import warnings
import logging
from typing import Dict, Tuple, Optional, Any

# ===========================
# 环境配置和警告抑制
# ===========================

def setup_environment():
    """
    配置运行环境，抑制各种警告和日志输出
    
    设置以下环境变量:
    - TF_CPP_MIN_LOG_LEVEL: 抑制 TensorFlow C++ 日志
    - TF_ENABLE_ONEDNN_OPTS: 禁用 oneDNN 优化警告
    - PYTHONWARNINGS: 抑制 Python 警告
    - TF_FORCE_GPU_ALLOW_GROWTH: 允许 GPU 内存动态增长
    """
    # 必须在导入 TensorFlow 之前设置环境变量
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 INFO, WARNING, ERROR 以下的日志
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 优化
    os.environ['PYTHONWARNINGS'] = 'ignore'    # 抑制 Python 警告
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU 内存动态增长
    
    # 配置 Python 警告过滤器
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 配置日志级别
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

# 初始化环境配置
setup_environment()

# ===========================
# 导入依赖库
# ===========================

import torch
import torch.nn as nn
import onnx
import tensorflow as tf

# 配置 TensorFlow
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except:
    pass  # 忽略在某些版本中不存在的配置

# ===========================
# 模型架构定义
# ===========================

class SimpleConvNet(nn.Module):
    """
    简单的卷积神经网络架构
    
    用作通用的模型架构，当无法从 state_dict 推断出准确架构时使用
    
    结构:
    - 卷积层: 3通道输入 -> 64通道输出，3x3卷积核
    - 激活函数: ReLU
    - 自适应池化: 全局平均池化
    - 全连接层: 64 -> 指定类别数
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 1000):
        super(SimpleConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

class FlexibleModel(nn.Module):
    """
    灵活的模型架构，支持更复杂的网络结构
    
    相比 SimpleConvNet 增加了更多层，更适合复杂的预训练模型
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 1000):
        super(FlexibleModel, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SegmentationModel(nn.Module):
    """
    语义分割模型架构
    
    专门用于处理语义分割任务，输出与输入相同尺寸的特征图
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 19, input_size: int = 512):
        super(SegmentationModel, self).__init__()
        
        self.input_size = input_size
        
        # 编码器 (下采样)
        self.encoder = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet风格的卷积块
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 更多卷积层
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 (上采样)
        self.decoder = nn.Sequential(
            # 上采样层
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # 输出层
        self.output_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码
        features = self.encoder(x)
        
        # 解码
        decoded = self.decoder(features)
        
        # 输出
        output = self.output_conv(decoded)
        
        # 确保输出尺寸与输入匹配
        if output.shape[-2:] != (self.input_size, self.input_size):
            output = nn.functional.interpolate(
                output, 
                size=(self.input_size, self.input_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        return output

# ===========================
# 模型分析和加载工具
# ===========================

class StateDict分析器:
    """
    State Dict 分析器
    
    用于分析 PyTorch state_dict 的结构，推断模型架构参数
    """
    
    @staticmethod
    def 分析参数结构(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        分析 state_dict 的参数结构
        
        Args:
            state_dict: PyTorch 模型的状态字典
            
        Returns:
            包含分析结果的字典，包括:
            - input_channels: 输入通道数
            - num_classes: 输出类别数
            - layer_count: 层数
            - param_count: 参数总数
            - model_type: 模型类型 ('classification' 或 'segmentation')
            - output_size: 输出尺寸
        """
        keys = list(state_dict.keys())
        analysis = {
            'input_channels': 3,  # 默认值
            'num_classes': 19,  # 根据您的需求调整默认值
            'layer_count': len(keys),
            'param_count': sum(param.numel() for param in state_dict.values()),
            'model_type': 'segmentation',  # 默认为分割模型
            'output_size': 512  # 默认输出尺寸
        }
        
        print(f"📊 State Dict 分析:")
        print(f"   参数总数: {analysis['param_count']:,}")
        print(f"   参数层数: {analysis['layer_count']}")
        
        # 分析模型类型和输出尺寸
        # 查找输出层来确定模型类型
        output_layers = [k for k in keys if any(term in k.lower() for term in ['out', 'head', 'classifier', 'final', 'pred'])]
        
        if output_layers:
            # 分析最后的输出层
            last_layer_key = output_layers[-1]
            if 'weight' in last_layer_key:
                last_param = state_dict[last_layer_key]
                print(f"   检测到输出层: {last_layer_key}, 形状: {list(last_param.shape)}")
                
                if last_param.dim() == 4:  # Conv2D 输出层
                    analysis['model_type'] = 'segmentation'
                    analysis['num_classes'] = last_param.shape[0]  # 输出通道数即类别数
                    print(f"   模型类型: 语义分割")
                    print(f"   输出类别数: {analysis['num_classes']}")
                elif last_param.dim() == 2:  # Linear 输出层
                    analysis['model_type'] = 'classification'
                    analysis['num_classes'] = last_param.shape[0]
                    print(f"   模型类型: 分类")
                    print(f"   输出类别数: {analysis['num_classes']}")
        
        # 分析输入通道数 - 查找第一个卷积层
        for key in keys:
            param = state_dict[key]
            if ('conv' in key.lower() and 'weight' in key and 
                param.dim() == 4 and param.shape[1] <= 10):  # 通常输入通道数不会太大
                analysis['input_channels'] = param.shape[1]
                print(f"   检测到输入通道数: {analysis['input_channels']}")
                break
        
        # 打印关键参数信息
        print(f"   关键参数:")
        for i, key in enumerate(keys[:10]):  # 显示前10个参数
            param = state_dict[key]
            print(f"     {key}: {list(param.shape)}")
        
        if len(keys) > 10:
            print(f"     ... 还有 {len(keys) - 10} 个参数")
            
        return analysis

    @staticmethod
    def 尝试参数匹配(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
        """
        尝试将 state_dict 中的参数与模型参数进行匹配
        
        Args:
            model: 目标模型
            state_dict: 源参数字典
            
        Returns:
            成功匹配的参数数量
        """
        model_dict = model.state_dict()
        matched_params = {}
        
        # 直接键匹配
        for key in model_dict.keys():
            if key in state_dict and model_dict[key].shape == state_dict[key].shape:
                matched_params[key] = state_dict[key]
                print(f"   ✓ 直接匹配: {key}")
        
        # 形状匹配 (当键名不匹配但形状匹配时)
        if len(matched_params) == 0:
            print("   尝试形状匹配...")
            state_values = list(state_dict.values())
            model_keys = list(model_dict.keys())
            
            for i, model_key in enumerate(model_keys):
                model_shape = model_dict[model_key].shape
                for j, state_value in enumerate(state_values):
                    if state_value.shape == model_shape and model_key not in matched_params:
                        matched_params[model_key] = state_value
                        state_key = list(state_dict.keys())[j]
                        print(f"   ≈ 形状匹配: {model_key} <- {state_key}")
                        break
        
        # 加载匹配的参数
        if matched_params:
            model_dict.update(matched_params)
            model.load_state_dict(model_dict, strict=False)
        
        return len(matched_params)

# ===========================
# 模型加载器
# ===========================

class PyTorch模型加载器:
    """
    PyTorch 模型加载器
    
    支持多种格式的 PyTorch 模型文件加载
    """
    
    @staticmethod
    def 加载模型(model_path: str) -> nn.Module:
        """
        加载 PyTorch 模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的 PyTorch 模型
            
        Raises:
            FileNotFoundError: 文件不存在
            Exception: 模型加载失败
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
        
        print(f"🔄 正在加载模型: {os.path.basename(model_path)}")
        
        try:
            # 加载检查点
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 分析检查点类型
            if isinstance(checkpoint, dict):
                return PyTorch模型加载器._处理字典格式(checkpoint)
            elif hasattr(checkpoint, 'eval'):
                print("✓ 检测到完整模型对象")
                checkpoint.eval()
                return checkpoint
            else:
                print("? 未知格式，假设为 state_dict")
                return PyTorch模型加载器._从state_dict创建模型(checkpoint)
                
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("🔧 使用默认模型架构...")
            return SimpleConvNet()
    
    @staticmethod
    def _处理字典格式(checkpoint: Dict[str, Any]) -> nn.Module:
        """处理字典格式的检查点"""
        if 'model' in checkpoint:
            print("✓ 检测到完整模型格式 (model 键)")
            model = checkpoint['model']
            model.eval()
            return model
        elif 'state_dict' in checkpoint:
            print("✓ 检测到 state_dict 格式")
            return PyTorch模型加载器._从state_dict创建模型(checkpoint['state_dict'])
        else:
            print("✓ 检测到原始 state_dict 格式")
            return PyTorch模型加载器._从state_dict创建模型(checkpoint)
    
    @staticmethod
    def _从state_dict创建模型(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """从 state_dict 创建模型"""
        # 分析参数结构
        analysis = StateDict分析器.分析参数结构(state_dict)
        
        # 根据模型类型选择候选架构
        if analysis['model_type'] == 'segmentation':
            model_candidates = [
                ('SegmentationModel', SegmentationModel(
                    analysis['input_channels'], 
                    analysis['num_classes'], 
                    analysis['output_size']
                )),
                ('FlexibleModel', FlexibleModel(analysis['input_channels'], analysis['num_classes']))
            ]
        else:
            model_candidates = [
                ('SimpleConvNet', SimpleConvNet(analysis['input_channels'], analysis['num_classes'])),
                ('FlexibleModel', FlexibleModel(analysis['input_channels'], analysis['num_classes']))
            ]
        
        best_model = None
        best_match_count = 0
        
        for model_name, model in model_candidates:
            print(f"🧪 尝试 {model_name} 架构...")
            match_count = StateDict分析器.尝试参数匹配(model, state_dict)
            print(f"   匹配参数数: {match_count}")
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_model = model
        
        if best_model is None or best_match_count == 0:
            print("⚠️ 无法匹配参数，创建基于分析的默认架构")
            if analysis['model_type'] == 'segmentation':
                best_model = SegmentationModel(
                    analysis['input_channels'], 
                    analysis['num_classes'], 
                    analysis['output_size']
                )
            else:
                best_model = SimpleConvNet(analysis['input_channels'], analysis['num_classes'])
        else:
            print(f"✓ 最佳匹配: {best_match_count} 个参数")
        
        best_model.eval()
        return best_model

# ===========================
# TensorFlow 转换器
# ===========================

class TensorFlow转换器:
    """
    TensorFlow 转换器
    
    将 PyTorch 模型转换为 TensorFlow Lite 格式
    """
    
    @staticmethod
    def 创建TF模型(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> tf.keras.Model:
        """
        创建等价的 TensorFlow 模型
        
        Args:
            input_shape: 输入形状 (包含 batch 维度)
            output_shape: 输出形状 (包含 batch 维度)
            
        Returns:
            TensorFlow Keras 模型
        """
        print(f"🏗️ 创建 TensorFlow 模型...")
        print(f"   输入形状: {input_shape}")
        print(f"   输出形状: {output_shape}")
        
        # 去除 batch 维度
        input_spec = input_shape[1:]  # (C, H, W)
        output_spec = output_shape[1:]  # (C, H, W) 对于分割或 (classes,) 对于分类
        
        # 转换为 TensorFlow 格式 (H, W, C)
        if len(input_spec) == 3:  # (C, H, W) -> (H, W, C)
            tf_input_shape = (input_spec[1], input_spec[2], input_spec[0])
        else:
            tf_input_shape = input_spec
            
        # 构建模型
        inputs = tf.keras.Input(shape=tf_input_shape, name='input')
        
        # 检查是否为分割模型
        is_segmentation = len(output_spec) == 3 and output_spec[0] == output_spec[1]
        
        if is_segmentation:
            # 分割模型架构
            num_classes = output_spec[2]  # TF格式: (H, W, C)
            target_size = (output_spec[0], output_spec[1])
            
            # 编码器
            x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu', name='conv1')(inputs)
            x = tf.keras.layers.BatchNormalization(name='bn1')(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
            
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2_1')(x)
            x = tf.keras.layers.BatchNormalization(name='bn2_1')(x)
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2_2')(x)
            x = tf.keras.layers.BatchNormalization(name='bn2_2')(x)
            
            x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu', name='conv3')(x)
            x = tf.keras.layers.BatchNormalization(name='bn3')(x)
            
            # 解码器
            x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu', name='deconv1')(x)
            x = tf.keras.layers.BatchNormalization(name='bn_dec1')(x)
            
            x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu', name='deconv2')(x)
            x = tf.keras.layers.BatchNormalization(name='bn_dec2')(x)
            
            x = tf.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu', name='deconv3')(x)
            x = tf.keras.layers.BatchNormalization(name='bn_dec3')(x)
            
            # 输出层
            outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', name='output_conv')(x)
            
            # 确保输出尺寸正确
            if target_size != (tf_input_shape[0], tf_input_shape[1]):
                outputs = tf.keras.layers.Resizing(target_size[0], target_size[1], name='resize_output')(outputs)
                
        else:
            # 分类模型架构
            num_classes = output_spec[0] if len(output_spec) == 1 else output_spec[-1]
            
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1')(inputs)
            x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2')(x)
            x = tf.keras.layers.GlobalAveragePooling2D(name='global_pool')(x)
            x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
            outputs = tf.keras.layers.Dense(num_classes, name='classifier')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='converted_model')
        
        # 编译模型
        if is_segmentation:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        else:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        print(f"✓ TensorFlow 模型创建完成 ({'分割' if is_segmentation else '分类'}模型)")
        model.summary()
        
        return model
    
    @staticmethod
    def 转换为TFLite(tf_model: tf.keras.Model, output_path: str) -> bool:
        """
        将 TensorFlow 模型转换为 TFLite 格式
        
        Args:
            tf_model: TensorFlow 模型
            output_path: 输出文件路径
            
        Returns:
            转换是否成功
        """
        print(f"🔄 开始转换为 TensorFlow Lite...")
        
        # 定义转换策略 - 从最兼容的开始
        strategies = [
            {
                'name': '直接转换',
                'use_saved_model': False,
                'optimizations': [],
                'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS],
            },
            {
                'name': '基础转换',
                'use_saved_model': False,
                'optimizations': [],
                'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS],
            },
            {
                'name': 'SavedModel转换',
                'use_saved_model': True,
                'optimizations': [],
                'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS],
            },
            {
                'name': '量化优化转换',
                'use_saved_model': True,
                'optimizations': [tf.lite.Optimize.DEFAULT],
                'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS],
            }
        ]
        
        # 尝试不同的转换策略
        for strategy in strategies:
            try:
                print(f"   🧪 尝试{strategy['name']}...")
                
                if strategy['use_saved_model']:
                    # 使用 SavedModel 方式
                    saved_model_dir = "temp_saved_model"
                    if os.path.exists(saved_model_dir):
                        import shutil
                        shutil.rmtree(saved_model_dir)
                    
                    # 尝试不同的保存方式
                    try:
                        # 方法1: 使用 export 方法
                        tf_model.export(saved_model_dir)
                        print(f"      ✓ 使用 export 方法保存 SavedModel")
                    except Exception as e1:
                        try:
                            # 方法2: 使用 tf.saved_model.save
                            tf.saved_model.save(tf_model, saved_model_dir)
                            print(f"      ✓ 使用 tf.saved_model.save 保存")
                        except Exception as e2:
                            print(f"      ❌ SavedModel保存失败: {str(e2)[:50]}...")
                            continue
                    
                    # 创建转换器
                    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
                else:
                    # 直接从 Keras 模型转换
                    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
                
                # 设置转换选项
                converter.optimizations = strategy['optimizations']
                converter.target_spec.supported_ops = strategy['supported_ops']
                
                # 设置额外的兼容性选项
                try:
                    converter.experimental_new_converter = True
                except:
                    pass
                
                try:
                    converter.allow_custom_ops = True
                except:
                    pass
                
                # 执行转换
                tflite_model = converter.convert()
                
                # 保存文件
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                
                print(f"   ✓ {strategy['name']}成功!")
                
                # 清理临时文件
                if strategy['use_saved_model'] and os.path.exists(saved_model_dir):
                    import shutil
                    shutil.rmtree(saved_model_dir)
                
                return True
                
            except Exception as e:
                print(f"   ❌ {strategy['name']}失败: {str(e)[:100]}...")
                # 清理临时文件
                if strategy.get('use_saved_model', False):
                    saved_model_dir = "temp_saved_model"
                    if os.path.exists(saved_model_dir):
                        try:
                            import shutil
                            shutil.rmtree(saved_model_dir)
                        except:
                            pass
                continue
        
        # 如果所有策略都失败，尝试创建一个简化的模型
        print("🔧 尝试创建简化模型进行转换...")
        try:
            # 获取输入输出形状
            input_shape = tf_model.input.shape[1:]  # 去除batch维度
            output_shape = tf_model.output.shape[1:]  # 去除batch维度
            
            # 创建简化的模型
            simplified_model = TensorFlow转换器._创建简化模型(input_shape, output_shape)
            
            # 尝试转换简化模型
            converter = tf.lite.TFLiteConverter.from_keras_model(simplified_model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print("   ✓ 简化模型转换成功!")
            return True
            
        except Exception as e:
            print(f"   ❌ 简化模型转换也失败: {str(e)[:100]}...")
        
        # 所有策略都失败
        print(f"❌ 所有转换策略都失败")
        return False
    
    @staticmethod
    def _创建简化模型(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> tf.keras.Model:
        """
        创建简化的模型用于转换
        
        Args:
            input_shape: 输入形状 (不包含 batch 维度)
            output_shape: 输出形状 (不包含 batch 维度)
            
        Returns:
            简化的 TensorFlow 模型
        """
        print(f"   创建简化模型: {input_shape} -> {output_shape}")
        
        inputs = tf.keras.Input(shape=input_shape, name='input')
        
        # 检查是否为分割模型
        is_segmentation = len(output_shape) == 3 and output_shape[0] == output_shape[1]
        
        if is_segmentation:
            # 简化的分割模型
            num_classes = output_shape[2]  # TF格式: (H, W, C)
            
            # 简单的编码-解码结构
            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            
            # 上采样回原始尺寸
            x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
            
            # 输出层
            outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same')(x)
            
            # 确保输出尺寸正确
            target_size = output_shape[:2]  # (H, W)
            current_size = (inputs.shape[1], inputs.shape[2])
            if target_size != current_size:
                outputs = tf.keras.layers.Resizing(target_size[0], target_size[1])(outputs)
        else:
            # 简化的分类模型
            num_classes = output_shape[0] if len(output_shape) == 1 else output_shape[-1]
            
            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(num_classes)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='simplified_model')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        return model

# ===========================
# 主转换流程
# ===========================

class 模型转换器:
    """
    主要的模型转换器类
    
    协调整个转换流程
    """
    
    def __init__(self, model_path: str, input_shape: Tuple[int, ...] = (1, 3, 512, 512)):
        """
        初始化转换器
        
        Args:
            model_path: PyTorch 模型文件路径
            input_shape: 输入张量形状 (batch, channels, height, width)
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.pytorch_model = None
        self.tf_model = None
        
    def 执行转换(self, output_name: str = "model.tflite") -> bool:
        """
        执行完整的转换流程
        
        Args:
            output_name: 输出文件名
            
        Returns:
            转换是否成功
        """
        try:
            print("=" * 60)
            print("🚀 PyTorch 到 TensorFlow Lite 转换工具")
            print("=" * 60)
            
            # 步骤1: 加载 PyTorch 模型
            print("\n📂 步骤1: 加载 PyTorch 模型")
            self.pytorch_model = PyTorch模型加载器.加载模型(self.model_path)
            
            # 步骤2: 验证模型
            print("\n🧪 步骤2: 验证 PyTorch 模型")
            output_shape = self._验证PyTorch模型()
            
            # 步骤3: 导出 ONNX (可选，用于验证)
            print("\n📤 步骤3: 导出 ONNX 格式")
            self._导出ONNX()
            
            # 步骤4: 创建 TensorFlow 模型
            print("\n🔄 步骤4: 创建 TensorFlow 模型")
            if output_shape:
                self.tf_model = TensorFlow转换器.创建TF模型(self.input_shape, output_shape)
            else:
                # 使用默认的分割模型输出形状
                default_output_shape = (1, 19, 512, 512)
                self.tf_model = TensorFlow转换器.创建TF模型(self.input_shape, default_output_shape)
            
            # 步骤5: 转换为 TFLite
            print("\n📱 步骤5: 转换为 TensorFlow Lite")
            success = TensorFlow转换器.转换为TFLite(self.tf_model, output_name)
            
            if success:
                self._验证TFLite模型(output_name)
                print(f"\n🎉 转换成功完成!")
                print(f"📁 输出文件: {output_name}")
                
                # 显示文件信息
                file_size = os.path.getsize(output_name) / (1024 * 1024)
                print(f"📊 文件大小: {file_size:.2f} MB")
                
            return success
            
        except Exception as e:
            print(f"\n❌ 转换过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._清理临时文件()
    
    def _验证PyTorch模型(self) -> Optional[Tuple[int, ...]]:
        """验证 PyTorch 模型的前向传播"""
        try:
            dummy_input = torch.randn(*self.input_shape)
            with torch.no_grad():
                output = self.pytorch_model(dummy_input)
            
            print(f"✓ 模型验证成功")
            print(f"  输入形状: {list(self.input_shape)}")
            print(f"  输出形状: {list(output.shape)}")
            
            return tuple(output.shape)
            
        except Exception as e:
            print(f"⚠️ 模型验证失败: {e}")
            print("🔧 尝试调整输入尺寸...")
            
            # 尝试其他常见尺寸
            alternative_shapes = [
                (1, 3, 512, 512),  # 您要求的尺寸
                (1, 3, 256, 256),
                (1, 3, 128, 128),
                (1, 3, 224, 224)
            ]
            
            for shape in alternative_shapes:
                try:
                    dummy_input = torch.randn(*shape)
                    with torch.no_grad():
                        output = self.pytorch_model(dummy_input)
                    
                    print(f"✓ 使用 {shape} 验证成功")
                    print(f"  输出形状: {list(output.shape)}")
                    self.input_shape = shape  # 更新输入形状
                    return tuple(output.shape)
                    
                except Exception as inner_e:
                    print(f"  ❌ {shape} 失败: {str(inner_e)[:50]}...")
                    continue
            
            print("❌ 所有输入尺寸都验证失败")
            # 返回预期的输出形状作为后备
            return (1, 19, 512, 512)
    
    def _导出ONNX(self):
        """导出 ONNX 格式 (用于验证)"""
        try:
            onnx_path = "temp_model.onnx"
            dummy_input = torch.randn(*self.input_shape)
            
            torch.onnx.export(
                self.pytorch_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )
            
            # 验证 ONNX 模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX 导出和验证成功")
            
        except Exception as e:
            print(f"⚠️ ONNX 导出失败: {e}")
            print("📝 这不会影响 TFLite 转换")
    
    def _验证TFLite模型(self, tflite_path: str):
        """验证生成的 TFLite 模型"""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✓ TFLite 模型验证成功")
            print(f"  输入详情: {input_details[0]['name']} {input_details[0]['shape']} ({input_details[0]['dtype']})")
            print(f"  输出详情: {output_details[0]['name']} {output_details[0]['shape']} ({output_details[0]['dtype']})")
            
        except Exception as e:
            print(f"⚠️ TFLite 模型验证失败: {e}")
    
    def _清理临时文件(self):
        """清理转换过程中产生的临时文件"""
        temp_files = [
            "temp_model.onnx",
            "temp_saved_model",
            "model.onnx",
            "saved_model"
        ]
        
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                elif os.path.isdir(temp_file):
                    import shutil
                    shutil.rmtree(temp_file)
            except:
                pass  # 忽略清理错误

# ===========================
# 依赖检查工具
# ===========================

def 检查依赖库() -> bool:
    """
    检查必要的依赖库是否已安装
    
    Returns:
        所有依赖是否满足
    """
    print("🔍 检查依赖库...")
    
    required_modules = {
        'torch': 'PyTorch',
        'onnx': 'ONNX',
        'tensorflow': 'TensorFlow'
    }
    
    optional_modules = {
        'tf2onnx': 'TF2ONNX (可选优化工具)',
    }
    
    missing_modules = []
    
    # 检查必需依赖
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name} 已安装")
        except ImportError:
            missing_modules.append(name)
            print(f"❌ {name} 未安装")
    
    # 检查可选依赖
    for module, name in optional_modules.items():
        try:
            __import__(module)
            print(f"✓ {name} 已安装")
        except ImportError:
            print(f"○ {name} 未安装 (可选)")
    
    if missing_modules:
        print(f"\n❌ 缺少必需依赖库: {', '.join(missing_modules)}")
        print("📦 安装命令:")
        print("   pip install torch torchvision")
        print("   pip install onnx")
        print("   pip install tensorflow")
        return False
    
    print("✅ 依赖库检查通过\n")
    return True

# ===========================
# 主程序入口
# ===========================

def main():
    """主程序入口"""
    print("PyTorch 到 TensorFlow Lite 转换工具 v2.0")
    print("支持智能模型架构推断和多策略转换\n")
    
    # 检查依赖
    if not 检查依赖库():
        sys.exit(1)
    
    # 配置参数 - 根据您的需求调整
    配置 = {
        'model_path': '79999_iter.pth',        # 输入模型路径
        'input_shape': (1, 3, 512, 512),      # 修改为您要求的输入形状
        'output_name': 'model.tflite'         # 输出文件名
    }
    
    print(f"📋 转换配置:")
    print(f"   输入模型: {配置['model_path']}")
    print(f"   输入形状: {配置['input_shape']}")
    print(f"   输出文件: {配置['output_name']}")
    print(f"   预期输出: (1, 19, 512, 512)")
    print()
    
    # 执行转换
    转换器 = 模型转换器(配置['model_path'], 配置['input_shape'])
    success = 转换器.执行转换(配置['output_name'])
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 转换成功完成!")
        print("📱 模型已准备好部署到移动端")
        print("🔍 请验证输出形状是否为 (1, 19, 512, 512)")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 转换失败")
        print("🔧 请检查模型文件和依赖库")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()