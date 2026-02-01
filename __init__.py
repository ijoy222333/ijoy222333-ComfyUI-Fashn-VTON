import sys
import subprocess
import importlib.util
import os

# --- 1. 定义自动安装函数 ---
def is_installed(package_name):
    try:
        return importlib.util.find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False

def install_package(package_name, install_name=None, no_deps=False):
    if install_name is None:
        install_name = package_name
    
    if is_installed(package_name):
        return True
        
    print(f"⏳ [Fashn-VTON] 检测到缺少关键库 '{package_name}'，正在自动安装...")
    
    command = [sys.executable, "-m", "pip", "install", install_name]
    if no_deps:
        command.append("--no-deps")
        
    try:
        subprocess.check_call(command)
        print(f"✅ [Fashn-VTON] '{package_name}' 安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [Fashn-VTON] 自动安装失败: {e}")
        return False

# --- 2. 启动时检查并安装依赖 ---
# 这里的 "fashn_human_parser" 是导入名，"fashn-human-parser" 是 pip 安装名
# no_deps=True 是为了保护用户的环境不被降级
install_package("fashn_human_parser", "fashn-human-parser", no_deps=True)

# 确保 onnxruntime-gpu 存在 (可选，根据你的需求决定是否强制帮用户装)
# if "onnxruntime_gpu" not in sys.modules and not is_installed("onnxruntime_gpu"):
#     install_package("onnxruntime_gpu", "onnxruntime-gpu")

# --- 3. 导入节点逻辑 ---
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']