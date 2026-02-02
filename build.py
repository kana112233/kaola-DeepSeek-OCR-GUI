#!/usr/bin/env python3
"""
PyInstaller build script for kaola-DeepSeek-OCR-GUI
"""
import os
import sys
import subprocess

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def build():
    """Build the executable with PyInstaller"""

    # Install PyInstaller if needed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Install all dependencies
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers==4.46.3", "pillow", "numpy",
        "addict", "einops", "easydict", "tiktoken",
        "fsspec", "huggingface-hub", "safetensors",
        "tokenizers", "regex", "requests", "tqdm",
        "matplotlib", "torchvision", "accelerate>=0.26.0"
    ])

    print("Building with PyInstaller...")

    name = "kaola-DeepSeek-OCR-GUI"

    # Platform-specific settings
    if sys.platform == "darwin":
        cmd = [
            "pyinstaller",
            "--name", name,
            "--onedir",
            "--windowed",
            "--noconfirm",
        ]
    else:
        cmd = [
            "pyinstaller",
            "--name", name,
            "--onefile",
            "--windowed",
            "--noconfirm",
        ]

    # Common parameters
    cmd += [
        # Core
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=tkinter",
        "--hidden-import=tkinter.scrolledtext",
        # PyTorch
        "--hidden-import=torch",
        "--hidden-import=torch.nn",
        "--hidden-import=torch.nn.functional",
        "--hidden-import=torchvision",
        "--hidden-import=torchvision.transforms",
        # Transformers - use collect-submodules to preserve structure
        "--collect-submodules=transformers",
        "--hidden-import=transformers",
        "--hidden-import=transformers.models",
        "--hidden-import=transformers.models.llama",
        "--hidden-import=transformers.generation",
        "--hidden-import=transformers.utils",
        "--hidden-import=tokenizers",
        "--hidden-import=huggingface_hub",
        # Data processing
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.pyplot",
        # Other dependencies
        "--hidden-import=addict",
        "--hidden-import=einops",
        "--hidden-import=easydict",
        "--hidden-import=tiktoken",
        "--hidden-import=fsspec",
        "--hidden-import=safetensors",
        "--hidden-import=regex",
        "--hidden-import=requests",
        "--hidden-import=tqdm",
        "--hidden-import=accelerate",
        # Collect packages
        "--collect-all=torch",
        "--collect-all=PIL",
        # Main script
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
