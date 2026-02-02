#!/usr/bin/env python3
"""
DeepSeek-OCR-2 GUI 程序
支持 Windows 和 Mac
"""
import os
import sys
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import torch


# ============================================
# 兼容性修复
# ============================================
def _setup_compatibility():
    """设置兼容性"""
    # 修复 .cuda() 硬编码
    if not hasattr(torch.Tensor, '_ocr_cuda_patched'):
        _original_cuda = torch.Tensor.cuda if hasattr(torch.Tensor, 'cuda') else None

        def _compat_cuda(self, device=None):
            if torch.cuda.is_available():
                return _original_cuda(self, device) if _original_cuda else self
            return self.to('cpu')

        torch.Tensor.cuda = _compat_cuda
        torch.Tensor._ocr_cuda_patched = True

        # 禁用 MPS（避免兼容性问题）
        torch.backends.mps.is_available = lambda: False


_setup_compatibility()


class DeepSeekOCR:
    """DeepSeek-OCR-2 核心类"""

    def __init__(self, model_path: str = ""):
        self.model_path = self._get_model_path(model_path)
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def _get_model_path(self, model_path: str) -> str:
        """获取模型路径"""
        if model_path and os.path.exists(model_path):
            return model_path

        common_paths = [
            os.path.join(os.path.dirname(__file__), "models", "DeepSeek-OCR-2"),
            os.path.expanduser("~/models/DeepSeek-OCR-2"),
            "./models/DeepSeek-OCR-2",
            "/Users/xiohu/work/ai-tools/models/DeepSeek-OCR-2",
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return "deepseek-ai/DeepSeek-OCR-2"

    def load(self, progress_callback=None):
        """加载模型"""
        with self._load_lock:
            if self._loaded:
                return

            if progress_callback:
                progress_callback("正在加载模型...")

            try:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=os.path.isdir(self.model_path)
                )

                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=os.path.isdir(self.model_path),
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=None
                )

                # 修补 bfloat16 问题
                _original_infer = self.model.infer

                def _patched_infer(*args, **kwargs):
                    original_bfloat16 = torch.bfloat16
                    torch.bfloat16 = torch.float32
                    try:
                        result = _original_infer(*args, **kwargs)
                    finally:
                        torch.bfloat16 = original_bfloat16
                    return result

                self.model.infer = _patched_infer

                # 转换到 CPU
                self.model = self.model.eval().cpu()
                for name, param in list(self.model.named_parameters()):
                    if param.dtype == torch.bfloat16:
                        param.data = param.data.to(torch.float32)

                self._loaded = True

                if progress_callback:
                    progress_callback("模型加载完成！")

            except Exception as e:
                if progress_callback:
                    progress_callback(f"模型加载失败: {e}")
                raise

    def ocr(self, image_path: str, mode: str = "free",
            progress_callback=None) -> str:
        """执行 OCR"""
        if not self._loaded:
            self.load(progress_callback)

        if progress_callback:
            progress_callback("正在识别...")

        if mode == "markdown":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        else:
            prompt = "<image>\nFree OCR. "

        with tempfile.TemporaryDirectory() as output_dir:
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=False
            )

            if progress_callback:
                progress_callback("识别完成！")

            return self._extract_text(result)

    def _extract_text(self, result) -> str:
        """提取文本结果"""
        if isinstance(result, dict):
            return result.get("text", result.get("result", str(result)))
        return str(result)


class OCRGUI:
    """OCR GUI 主窗口"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("kaola-DeepSeek-OCR-GUI")
        self.root.geometry("800x600")

        # Mac 优化
        if sys.platform == "darwin":
            self.root.createcommand("::tk::mac::ShowPreferences", lambda: None)

        self.ocr = None
        self.current_image_path = None
        self.preview_image = None

        self._setup_ui()
        self._load_model_async()

    def _setup_ui(self):
        """设置 UI"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill=tk.X)

        # 选择图像按钮
        ttk.Button(toolbar, text="选择图像", command=self._select_image).pack(side=tk.LEFT, padx=5)

        # 模式选择
        ttk.Label(toolbar, text="模式:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="free")
        mode_combo = ttk.Combobox(toolbar, textvariable=self.mode_var,
                                  values=["free", "markdown"], state="readonly", width=10)
        mode_combo.pack(side=tk.LEFT, padx=5)

        # 识别按钮
        self.ocr_button = ttk.Button(toolbar, text="开始识别", command=self._start_ocr, state=tk.DISABLED)
        self.ocr_button.pack(side=tk.LEFT, padx=5)

        # 状态标签
        self.status_label = ttk.Label(toolbar, text="正在加载模型...", foreground="orange")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # 主内容区
        content = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：图像预览
        left_frame = ttk.Frame(content)
        content.add(left_frame, weight=1)

        ttk.Label(left_frame, text="图像预览").pack(pady=5)

        self.image_label = ttk.Label(left_frame, text="请选择图像", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 右侧：识别结果
        right_frame = ttk.Frame(content)
        content.add(right_frame, weight=1)

        ttk.Label(right_frame, text="识别结果").pack(pady=5)

        self.result_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Courier", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 底部工具栏
        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill=tk.X)

        ttk.Button(bottom, text="复制结果", command=self._copy_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom, text="保存结果", command=self._save_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom, text="清空", command=self._clear).pack(side=tk.LEFT, padx=5)

    def _load_model_async(self):
        """异步加载模型"""
        def load_model():
            try:
                self.ocr = DeepSeekOCR()
                self.ocr.load(lambda msg: self._update_status(msg, "orange" if "加载" in msg else "green"))
            except Exception as e:
                self._update_status(f"模型加载失败: {e}", "red")

        threading.Thread(target=load_model, daemon=True).start()

    def _update_status(self, message: str, color: str = "black"):
        """更新状态"""
        def update():
            self.status_label.config(text=message, foreground=color)
            if "加载完成" in message:
                self.ocr_button.config(state=tk.NORMAL)

        self.root.after(0, update)

    def _select_image(self):
        """选择图像"""
        path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("所有文件", "*.*")
            ]
        )

        if path:
            self.current_image_path = path
            self._show_preview(path)
            self.ocr_button.config(state=tk.NORMAL)

    def _show_preview(self, path: str):
        """显示预览"""
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))

            if sys.platform == "darwin":
                photo = ImageTk.PhotoImage(img)
            else:
                photo = ImageTk.PhotoImage(img)

            self.preview_image = photo
            self.image_label.config(image=photo, text="")
        except Exception as e:
            self.image_label.config(text=f"预览失败: {e}", image="")

    def _start_ocr(self):
        """开始 OCR"""
        if not self.current_image_path or not self.ocr:
            return

        self.ocr_button.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)
        self._update_status("正在识别...", "orange")

        def do_ocr():
            try:
                result = self.ocr.ocr(
                    self.current_image_path,
                    mode=self.mode_var.get(),
                    progress_callback=lambda msg: self._update_status(msg, "orange")
                )

                def show_result():
                    self.result_text.insert(1.0, result)
                    self._update_status("识别完成！", "green")
                    self.ocr_button.config(state=tk.NORMAL)

                self.root.after(0, show_result)

            except Exception as e:
                def show_error():
                    self.result_text.insert(1.0, f"识别失败: {e}")
                    self._update_status("识别失败", "red")
                    self.ocr_button.config(state=tk.NORMAL)

                self.root.after(0, show_error)

        threading.Thread(target=do_ocr, daemon=True).start()

    def _copy_result(self):
        """复制结果"""
        text = self.result_text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._update_status("已复制到剪贴板", "green")

    def _save_result(self):
        """保存结果"""
        text = self.result_text.get(1.0, tk.END).strip()
        if not text:
            return

        path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".txt",
            filetypes=[
                ("文本文件", "*.txt"),
                ("Markdown", "*.md"),
                ("所有文件", "*.*")
            ]
        )

        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            self._update_status(f"已保存: {path}", "green")

    def _clear(self):
        """清空"""
        self.result_text.delete(1.0, tk.END)
        self.image_label.config(image="", text="请选择图像")
        self.current_image_path = None
        self.preview_image = None
        self.ocr_button.config(state=tk.DISABLED)
        self._update_status("已清空", "black")

    def run(self):
        """运行主循环"""
        self.root.mainloop()


def main():
    """主函数"""
    app = OCRGUI()
    app.run()


if __name__ == "__main__":
    main()
