# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for kaola-DeepSeek-OCR-GUI
Run: pyinstaller kaola-DeepSeek-OCR-GUI.spec
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules
import sys

datas = []
binaries = []
hiddenimports = [
    'PIL._tkinter_finder',
    'transformers',
    'transformers.models',
    'transformers.models.llama',
    'transformers.generation',
    'transformers.utils',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.transforms',
    'numpy',
    'PIL',
    'matplotlib',
    'matplotlib.pyplot',
    'tkinter',
    'tkinter.scrolledtext',
    'addict',
    'einops',
    'easydict',
    'tiktoken',
    'fsspec',
    'huggingface_hub',
    'safetensors',
    'tokenizers',
    'regex',
    'requests',
    'tqdm',
    'accelerate',
]

# Use collect_submodules for transformers to preserve directory structure
tmp_ret = collect_submodules('transformers')
hiddenimports += tmp_ret

# Use collect_all for torch and PIL
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('huggingface_hub')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['deepseek_ocr_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='kaola-DeepSeek-OCR-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Create .app bundle on macOS only
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name='kaola-DeepSeek-OCR-GUI.app',
        icon=None,
        bundle_identifier=None,
    )
