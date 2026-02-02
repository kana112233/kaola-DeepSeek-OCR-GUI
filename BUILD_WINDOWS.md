# Windows 打包说明

## 在 Windows 上打包

由于 PyInstaller 不支持跨平台打包，需要在 Windows 系统上打包 Windows 版本。

### 步骤

1. **安装 Python**

   下载并安装 Python 3.10+：https://www.python.org/downloads/

2. **克隆项目**

   ```cmd
   git clone https://github.com/kana112233/kaola-DeepSeek-OCR-GUI.git
   cd kaola-DeepSeek-OCR-GUI
   ```

3. **安装依赖**

   ```cmd
   pip install torch transformers pillow pyinstaller
   pip install addict einops easydict tiktoken
   ```

4. **运行打包脚本**

   ```cmd
   python build.py
   ```

5. **获取可执行文件**

   打包完成后，在 `dist` 目录下会生成：
   - `kaola-DeepSeek-OCR-GUI.exe` - Windows 可执行文件

### 分发

打包后的文件可以直接分发，用户无需安装 Python 环境。

### 注意事项

- Windows Defender 可能会误报，需要添加信任
- 首次运行可能需要几分钟加载模型
- 需要 Windows 10 或更高版本
