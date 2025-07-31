"""
自动安装PyTorch到TensorFlow Lite转换所需的依赖库
"""
import subprocess
import sys
import importlib
import warnings
import os

# 抑制 TensorFlow 和 protobuf 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

def check_module(module_name):
    """检查模块是否已安装"""
    try:
        # 特殊处理 onnx_tf
        if module_name == 'onnx_tf':
            # 尝试多种导入方式
            try:
                importlib.import_module('onnx_tf')
                return True
            except ImportError:
                try:
                    importlib.import_module('onnx_tf.frontend')
                    return True
                except ImportError:
                    # 最后尝试检查包是否通过pip安装
                    try:
                        from importlib.metadata import distribution
                        distribution('onnx-tf')
                        return True
                    except ImportError:
                        # 兼容 Python < 3.8
                        import pkg_resources
                        try:
                            pkg_resources.get_distribution('onnx-tf')
                            return True
                        except pkg_resources.DistributionNotFound:
                            return False
                    except Exception:
                        return False
        else:
            importlib.import_module(module_name)
            return True
    except ImportError:
        return False

def install_package(package):
    """安装指定包"""
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package} 安装失败: {e}")
        return False

def main():
    """主安装流程"""
    print("开始检查和安装依赖库...")
    
    # 定义依赖库映射 (模块名: 包名)
    dependencies = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'onnx': 'onnx',
        'onnx_tf': 'onnx-tf',
        'tensorflow': 'tensorflow',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    # 检查已安装的模块
    print("\n=== 检查依赖库 ===")
    for module, package in dependencies.items():
        if check_module(module):
            print(f"✓ {module} 已安装")
        else:
            print(f"✗ {module} 未安装")
            missing_packages.append(package)
    
    if not missing_packages:
        print("\n🎉 所有依赖库都已安装!")
        return True
    
    # 安装缺失的包
    print(f"\n=== 安装缺失的库 ===")
    print(f"需要安装: {', '.join(missing_packages)}")
    
    # 升级pip
    print("首先升级pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # 安装依赖
    failed_packages = []
    for package in missing_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  以下包安装失败: {', '.join(failed_packages)}")
        print("请手动安装或检查网络连接")
        return False
    else:
        print("\n🎉 所有依赖库安装完成!")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
