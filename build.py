#!/usr/bin/env python3
"""
PyInstaller 打包脚本
将 kaola-DeepSeek-OCR-GUI 打包成独立可执行程序
"""
import os
import sys
import subprocess

# 修复 Windows 编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def build():
    """打包程序"""

    # 确保安装了 PyInstaller
    try:
        import PyInstaller
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "pyinstaller"
        ])

    # 确保安装了所有依赖
    print("检查依赖...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers", "pillow", "numpy",
        "addict", "einops", "easydict", "tiktoken",
        "fsspec", "huggingface-hub", "safetensors",
        "tokenizers", "regex", "requests", "tqdm"
    ])

    print("开始打包...")

    # 确定输出名称
    if sys.platform == "darwin":
        name = "kaola-DeepSeek-OCR-GUI"
    elif sys.platform == "win32":
        name = "kaola-DeepSeek-OCR-GUI"
    else:
        name = "kaola-DeepSeek-OCR-GUI"

    # PyInstaller 命令
    cmd = [
        "pyinstaller",
        f"--name={name}",
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=transformers",
        "--hidden-import=torch",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "--hidden-import=tkinter",
        "--hidden-import=tkinter.scrolledtext",
        "--hidden-import=addict",
        "--hidden-import=einops",
        "--hidden-import=easydict",
        "--hidden-import=tiktoken",
        "--hidden-import=fsspec",
        "--hidden-import=huggingface_hub",
        "--hidden-import=safetensors",
        "--hidden-import=tokenizers",
        "--hidden-import=regex",
        "--hidden-import=requests",
        "--hidden-import=tqdm",
        "--hidden-import=transformers.models",
        "--hidden-import=transformers.generation",
        "--hidden-import=transformers.utils",
        "--hidden-import=torch.nn",
        "--hidden-import=torch.nn.functional",
        "--collect-all=transformers",
        "--collect-all=torch",
        "--collect-all=PIL",
        "--collect-all=huggingface_hub",
        "deepseek_ocr_gui.py"
    ]

    print(f"执行命令: {' '.join(cmd)}")

    subprocess.check_call(cmd)

    print("\n" + "=" * 60)
    print("打包完成！")
    print("=" * 60)
    print(f"可执行文件位于: dist/{name}")

    if sys.platform == "darwin":
        print(f"\nMac 版本: dist/{name}")
        print(f"运行: ./dist/{name}")
    elif sys.platform == "win32":
        print(f"\nWindows 版本: dist\\{name}.exe")
        print(f"运行: dist\\{name}.exe")
    else:
        print(f"\nLinux 版本: dist/{name}")


if __name__ == "__main__":
    build()
