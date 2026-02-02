# kaola-DeepSeek-OCR-GUI

DeepSeek-OCR-2 的独立 GUI 程序，支持 Windows 和 Mac。

## 功能

- 图像文字识别（OCR）
- 文档转 Markdown
- 图像预览
- 结果复制和保存

## 依赖

```bash
pip install torch transformers pillow
```

推荐版本：
```
transformers==4.46.3
torch>=2.0
pillow
```

## 使用方法

### 方式1：直接运行 Python 脚本

```bash
python deepseek_ocr_gui.py
```

### 方式2：打包成可执行程序

```bash
# 安装 PyInstaller
pip install pyinstaller

# 打包
python build.py

# 运行打包后的程序
# Mac:
./dist/kaola-DeepSeek-OCR-GUI

# Windows:
dist\kaola-DeepSeek-OCR-GUI.exe
```

## 模型配置

程序会自动在以下路径查找模型：

1. `./models/DeepSeek-OCR-2`
2. `~/models/DeepSeek-OCR-2`
3. HuggingFace: `deepseek-ai/DeepSeek-OCR-2`

下载模型：

```bash
# 使用 huggingface-cli
pip install huggingface-hub
huggingface-cli download deepseek-ai/DeepSeek-OCR-2 --local-dir ./models/DeepSeek-OCR-2

# 或使用 git-lfs
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 ./models/DeepSeek-OCR-2
```

## 使用说明

1. 点击"选择图像"按钮选择要识别的图像
2. 选择识别模式（Free OCR 或 Markdown）
3. 点击"开始识别"按钮
4. 等待识别完成，结果会显示在右侧
5. 可以复制结果或保存到文件

## 命令行版本

如果不需要 GUI，可以使用命令行版本：

```bash
python deepseek_ocr.py image.jpg

# 文档转 Markdown
python deepseek_ocr.py document.jpg --mode markdown

# 保存结果
python deepseek_ocr.py image.jpg -o result.txt
```

## 注意事项

- 首次运行需要加载模型，可能需要几分钟
- CPU 模式下识别速度较慢
- 建议使用 GPU（CUDA）以获得更好的性能

## 许可证

Apache 2.0
