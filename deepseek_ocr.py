#!/usr/bin/env python3
"""
DeepSeek-OCR-2 独立程序
支持图像文字识别和文档转 Markdown
"""
import os
import sys
import argparse
import tempfile
import torch
import numpy as np
from PIL import Image
from typing import Optional


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
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return self.to('mps')
            return self.to('cpu')

        torch.Tensor.cuda = _compat_cuda
        torch.Tensor._ocr_cuda_patched = True


_setup_compatibility()


class DeepSeekOCR:
    """DeepSeek-OCR-2 独立程序"""

    def __init__(self, model_path: str = "", device: str = "auto"):
        self.model_path = self._get_model_path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _get_model_path(self, model_path: str) -> str:
        """获取模型路径"""
        if model_path and os.path.exists(model_path):
            return model_path

        common_paths = [
            "/Users/xiohu/work/ai-tools/models/DeepSeek-OCR-2",
            os.path.expanduser("~/models/DeepSeek-OCR-2"),
            "./models/DeepSeek-OCR-2",
            "deepseek-ai/DeepSeek-OCR-2",
        ]

        for path in common_paths:
            if os.path.exists(path):
                print(f"[DeepSeek-OCR] Using local model: {path}")
                return path

        return "deepseek-ai/DeepSeek-OCR-2"

    def load(self):
        """加载模型"""
        if self._loaded:
            return

        print(f"[DeepSeek-OCR] Loading model from: {self.model_path}")

        # 确定设备
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device

        # CPU 模式禁用 MPS
        original_mps_available = None
        if device == "cpu":
            original_mps_available = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False

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
            if hasattr(self.model, 'infer'):
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

            print("[DeepSeek-OCR] Model loaded")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

        finally:
            if original_mps_available:
                torch.backends.mps.is_available = original_mps_available

        # 转换模型到目标设备
        self.model = self.model.eval()

        if device == "cpu":
            print("[DeepSeek-OCR] Converting model to CPU float32...")
            self.model = self.model.cpu()
            for name, param in list(self.model.named_parameters()):
                if param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float32)

        elif device == "mps":
            print("[DeepSeek-OCR] Converting model to MPS float16...")
            for module in self.model.modules():
                for name, param in list(module.named_parameters(recurse=False)):
                    if param.dtype != torch.float16:
                        param.data = param.data.to(torch.float16)
                for name, buffer in list(module.named_buffers(recurse=False)):
                    if buffer.dtype not in [torch.float16, torch.bool, torch.long]:
                        buffer.data = buffer.data.to(torch.float16)
            self.model = self.model.to('mps')

        if device in ["cuda", "mps"]:
            self.model = self.model.to(device)

        print(f"[DeepSeek-OCR] Model on {device}")
        self._loaded = True

    def ocr(
        self,
        image_path: str,
        prompt: str = "<image>\nFree OCR. ",
        base_size: int = 1024,
        image_size: int = 768,
        crop_mode: bool = True,
    ) -> str:
        """
        执行 OCR 识别

        Args:
            image_path: 图像文件路径
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            crop_mode: 是否使用裁剪模式

        Returns:
            识别结果文本
        """
        if not self._loaded:
            self.load()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with tempfile.TemporaryDirectory() as output_dir:
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False
            )

            return self._extract_text(result)

    def _extract_text(self, result) -> str:
        """提取文本结果"""
        if isinstance(result, dict):
            return result.get("text", result.get("result", str(result)))
        elif isinstance(result, (list, tuple)):
            return "\n".join(str(r) for r in result)
        return str(result)


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR-2: 文字识别和文档转 Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # Free OCR 模式
  python deepseek_ocr.py image.jpg

  # 文档转 Markdown
  python deepseek_ocr.py document.jpg --mode markdown

  # 使用自定义模型路径
  python deepseek_ocr.py image.jpg --model-path /path/to/model

  # 指定输出文件
  python deepseek_ocr.py image.jpg -o result.txt
        """
    )

    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("--model-path", "-m", default="", help="模型路径（本地或 HuggingFace）")
    parser.add_argument("--mode", choices=["free", "markdown"], default="free",
                        help="OCR 模式: free(自由识别) 或 markdown(文档转MD)")
    parser.add_argument("--base-size", type=int, default=1024, help="基础尺寸 (默认: 1024)")
    parser.add_argument("--image-size", type=int, default=768, help="图像尺寸 (默认: 768)")
    parser.add_argument("--no-crop", action="store_true", help="禁用裁剪模式")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="计算设备 (默认: auto)")
    parser.add_argument("--output", "-o", help="输出文件路径（可选）")

    args = parser.parse_args()

    # 确定提示词
    if args.mode == "markdown":
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    else:
        prompt = "<image>\nFree OCR. "

    # 创建 OCR 实例
    ocr = DeepSeekOCR(model_path=args.model_path, device=args.device)

    try:
        # 执行 OCR
        print(f"[DeepSeek-OCR] Processing: {args.image}")
        result = ocr.ocr(
            image_path=args.image,
            prompt=prompt,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=not args.no_crop,
        )

        # 输出结果
        print("\n" + "=" * 60)
        print("识别结果:")
        print("=" * 60)
        print(result)
        print("=" * 60)

        # 保存到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n[DeepSeek-OCR] 结果已保存到: {args.output}")

    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
