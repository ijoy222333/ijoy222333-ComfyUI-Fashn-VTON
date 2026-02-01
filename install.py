import sys
import subprocess
import os

def install_package(package_name, no_deps=False):
    """
    智能安装函数
    """
    try:
        # 构造 pip 安装命令
        cmd = [sys.executable, "-m", "pip", "install", package_name]
        
        # 如果指定了忽略依赖，加上参数
        if no_deps:
            cmd.append("--no-deps")
            
        print(f"Installing {package_name}...")
        # 执行命令
        subprocess.check_call(cmd)
        print(f"✅ {package_name} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

# --- 主逻辑 ---
try:
    import fashn_human_parser
    print("✅ fashn-human-parser is already installed.")
except ImportError:
    print("⏳ fashn-human-parser not found. Installing...")
    # 关键点：这里加上 no_deps=True，强制跳过依赖检查
    install_package("fashn-human-parser", no_deps=True)