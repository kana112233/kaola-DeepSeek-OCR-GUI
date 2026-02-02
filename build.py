#!/usr/bin/env python3
"""
PyInstaller 打包脚本
将 DeepSeek-OCR-2 打包成独立可执行程序
"""
import os
import sys
import subprocess
import shutil


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

    print("开始打包...")

    # PyInstaller 命令
    cmd = [
        "pyinstaller",
        "--name=DeepSeek-OCR-2",
        "--onefile",
        "--windowed",
        "--icon=icon.ico" if os.path.exists("icon.ico") else "",
        "--add-data=README.md:.",
        "--hidden-import=PIL._tkinter_finder",
        "--collect-all=transformers",
        "--collect-all=torch",
        "deepseek_ocr_gui.py"
    ]

    # 过滤空参数
    cmd = [x for x in cmd if x]

    print(f"执行命令: {' '.join(cmd)}")

    subprocess.check_call(cmd)

    print("\n打包完成！")
    print(f"可执行文件位于: dist/DeepSeek-OCR-2")
    if sys.platform == "darwin":
        print(f"Mac 版本: dist/DeepSeek-OCR-2.app")
    elif sys.platform == "win32":
        print(f"Windows 版本: dist/DeepSeek-OCR-2.exe")
    else:
        print(f"Linux 版本: dist/DeepSeek-OCR-2")


if __name__ == "__main__":
    build()
