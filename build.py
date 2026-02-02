#!/usr/bin/env python3
"""
Nuitka build script for kaola-DeepSeek-OCR-GUI
Nuitka compiles Python to C, then to binary - more reliable than PyInstaller
"""
import os
import sys
import subprocess
import shutil

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def build():
    """Build the executable with Nuitka"""

    # Install Nuitka if needed
    try:
        import nuitka
    except ImportError:
        print("Installing Nuitka...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka"])

    # Install all dependencies
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers==4.46.3", "pillow", "numpy",
        "addict", "einops", "easydict", "tiktoken",
        "fsspec", "huggingface-hub", "safetensors",
        "tokenizers", "regex", "requests", "tqdm",
        "matplotlib", "torchvision"
    ])

    print("Building with Nuitka...")

    name = "kaola-DeepSeek-OCR-GUI"

    # Nuitka command - automatically detects all dependencies
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--enable-plugin=tkinter",
        "--disable-console",
        "--output-dir=dist",
        "--output-filename=" + ("kaola-DeepSeek-OCR-GUI.app" if sys.platform == "darwin" else "kaola-DeepSeek-OCR-GUI.exe"),
        "--remove-output",  # Clean build files
        # Include data directories
        "--include-data-dir=../models/DeepSeek-OCR-2=models/DeepSeek-OCR-2",
        "deepseek_ocr_gui.py"
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)

    if sys.platform == "darwin":
        print(f"\nmacOS: dist/{name}.app")
        print(f"Run: open dist/{name}.app")
    elif sys.platform == "win32":
        print(f"\nWindows: dist\\{name}.exe")
        print(f"Run: dist\\{name}.exe")


if __name__ == "__main__":
    build()
