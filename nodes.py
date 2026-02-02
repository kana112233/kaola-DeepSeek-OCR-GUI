import os
import sys
import tempfile
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple

# 禁用 MPS（因为 DeepSeek-OCR-2 模型与 MPS 不兼容）
# 用户可以在 model_path 中指定本地模型路径
torch.backends.mps.is_available = lambda: False

# ============================================
# Transformers 版本兼容性修复
# 在模块导入时立即应用
# ============================================
def _setup_compatibility():
    """设置 transformers 版本兼容性"""

    # 1. 修复 LlamaFlashAttention2
    try:
        from transformers.models.llama import modeling_llama
        if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
            # 创建一个兼容类
            if hasattr(modeling_llama, 'LlamaAttention'):
                class LlamaFlashAttention2(modeling_llama.LlamaAttention):
                    pass
                modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2
    except Exception:
        pass

    # 2. 修复 is_torch_fx_available
    try:
        from transformers.utils import import_utils
        if not hasattr(import_utils, 'is_torch_fx_available'):
            import_utils.is_torch_fx_available = lambda: False
    except Exception:
        pass

    # 3. 修复 .cuda() 硬编码
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

    # 4. 修复 DynamicCache.from_legacy_cache 兼容性问题
    try:
        from transformers.generation import DynamicCache
        if not hasattr(DynamicCache, 'from_legacy_cache'):
            @classmethod
            def from_legacy_cache(cls, legacy_cache):
                """将旧格式的缓存转换为新的 DynamicCache 格式"""
                if legacy_cache is None:
                    return cls()

                cache = cls()
                for layer_cache in legacy_cache:
                    if layer_cache is None:
                        continue
                    # layer_cache 是 (key, value) 元组
                    key_states, value_states = layer_cache
                    cache.update(key_states, value_states)
                return cache

            DynamicCache.from_legacy_cache = from_legacy_cache
    except Exception:
        pass


_setup_compatibility()


class DeepSeekOCRModel:
    """DeepSeek-OCR-2 模型加载器"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self, device: str = "auto", dtype: Optional[torch.dtype] = None):
        if self._is_loaded:
            return

        print(f"[DeepSeek-OCR] Loading model from: {self.model_path}")

        # 确定设备
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # 确定数据类型
        if dtype is None:
            # 对于 CPU，使用 float32 并转换 bfloat16 权重
            if device == "cpu":
                dtype = torch.float32
            elif device == "cuda":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

        # CPU 模式下禁用 MPS 自动检测
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

            # 加载模型 - CPU 模式强制使用 float32
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=os.path.isdir(self.model_path),
                torch_dtype=torch.float32,  # 强制使用 float32
                low_cpu_mem_usage=True,
                device_map=None
            )

            # 修改模型配置，禁用 bfloat16
            if hasattr(self.model, 'config'):
                self.model.config.use_bfloat16 = False
                if hasattr(self.model.config, 'torch_dtype'):
                    self.model.config.torch_dtype = torch.float32

            # 递归修改所有子模块的配置
            for name, module in self.model.named_modules():
                if hasattr(module, 'config'):
                    module.config.use_bfloat16 = False
                    if hasattr(module.config, 'torch_dtype'):
                        module.config.torch_dtype = torch.float32

            print("[DeepSeek-OCR] Model loaded")

            # 修补模型的 infer 方法，防止 bfloat16 问题
            if hasattr(self.model, 'infer'):
                _original_infer = self.model.infer

                def _patched_infer(tokenizer, prompt='', image_file='', output_path='',
                                   base_size=1024, image_size=640, crop_mode=True,
                                   test_compress=False, save_results=False, eval_mode=False):
                    # 在 CPU 模式下，使用 float32 而不是 bfloat16
                    import builtins
                    original_bfloat16 = torch.bfloat16

                    # 临时替换 torch.bfloat16
                    def _fake_bfloat16(*args, **kwargs):
                        return torch.float32

                    torch.bfloat16 = torch.float32

                    try:
                        result = _original_infer(
                            tokenizer=tokenizer,
                            prompt=prompt,
                            image_file=image_file,
                            output_path=output_path,
                            base_size=base_size,
                            image_size=image_size,
                            crop_mode=crop_mode,
                            test_compress=test_compress,
                            save_results=save_results,
                            eval_mode=eval_mode
                        )
                    finally:
                        torch.bfloat16 = original_bfloat16

                    return result

                self.model.infer = _patched_infer
                print("[DeepSeek-OCR] Applied bfloat16 compatibility patch")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

        finally:
            if original_mps_available:
                torch.backends.mps.is_available = original_mps_available

        # 转换模型到目标设备
        self.model = self.model.eval()

        # 确保模型在正确的设备上
        if device == "cpu":
            # 强制所有模块到 CPU，并转换 bfloat16 到 float32
            print("[DeepSeek-OCR] Moving all modules to CPU and converting bfloat16 to float32...")
            self.model = self.model.cpu()

            # 递归转换所有参数和缓冲区到 float32
            converted_count = 0
            for module_name, module in self.model.named_modules():
                for param_name, param in list(module.named_parameters(recurse=False)):
                    if param.dtype == torch.bfloat16:
                        param.data = param.data.to(torch.float32)
                        converted_count += 1
                for buffer_name, buffer in list(module.named_buffers(recurse=False)):
                    if buffer.dtype == torch.bfloat16:
                        buffer.data = buffer.data.to(torch.float32)
                        converted_count += 1
            print(f"[DeepSeek-OCR] Converted {converted_count} tensors to float32")
        elif device == "mps":
            print("[DeepSeek-OCR] Converting all tensors to float16 for MPS...")

            # 转换所有参数和缓冲区
            for module in self.model.modules():
                for name, param in list(module.named_parameters(recurse=False)):
                    if param.dtype != torch.float16:
                        param.data = param.data.to(torch.float16)
                for name, buffer in list(module.named_buffers(recurse=False)):
                    if buffer.dtype != torch.float16 and buffer.dtype != torch.bool and buffer.dtype != torch.long:
                        buffer.data = buffer.data.to(torch.float16)

            # 修改配置，禁用 bfloat16
            if hasattr(self.model, 'config'):
                self.model.config.use_bfloat16 = False
                if hasattr(self.model.config, 'torch_dtype'):
                    self.model.config.torch_dtype = torch.float16

            # 递归修改所有子模块
            for name, module in self.model.named_modules():
                if hasattr(module, 'config'):
                    module.config.use_bfloat16 = False
                    if hasattr(module.config, 'torch_dtype'):
                        module.config.torch_dtype = torch.float16

            self.model = self.model.to('mps')

        if device in ["cuda", "mps"]:
            self.model = self.model.to(device)

        print(f"[DeepSeek-OCR] Model on {device} with {dtype}")
        self._is_loaded = True


class DeepSeekOCRNode:
    """DeepSeek-OCR-2 节点"""

    def __init__(self):
        self.model_loader = None
        self.default_model_path = "deepseek-ai/DeepSeek-OCR-2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "<image>\n<|grounding|>Convert the document to markdown. ",
                    "multiline": True,
                }),
                "base_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "image_size": ("INT", {"default": 768, "min": 384, "max": 1536, "step": 64}),
                "crop_mode": ("BOOLEAN", {"default": True}),
                "model_path": ("STRING", {"default": "", "display": "prompt"}),
            },
            "optional": {
                "device": ("STRING", {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ocr_process"
    CATEGORY = "DeepSeek"

    def _get_model_path(self, model_path: str) -> str:
        if model_path and os.path.exists(model_path):
            return model_path

        common_paths = [
            "/Users/xiohu/work/ai-tools/models/DeepSeek-OCR-2",
            os.path.expanduser("~/models/DeepSeek-OCR-2"),
            "./models/DeepSeek-OCR-2",
        ]

        for path in common_paths:
            if os.path.exists(path):
                print(f"[DeepSeek-OCR] Using local model: {path}")
                return path

        return self.default_model_path

    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]

        image_np = image_tensor.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        return Image.fromarray(image_np)

    def ocr_process(self, image: torch.Tensor, prompt: str, base_size: int,
                    image_size: int, crop_mode: bool, model_path: str,
                    device: str = "auto") -> Tuple[str]:
        actual_model_path = self._get_model_path(model_path)

        if self.model_loader is None or self.model_loader.model_path != actual_model_path:
            self.model_loader = DeepSeekOCRModel(actual_model_path)
            self.model_loader.load(device=device)

        pil_image = self.tensor_to_pil(image)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path)

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # MPS: 在推理前确保所有张量类型一致，并打印问题模块
                device_type = next(self.model_loader.model.parameters()).device.type
                if device_type == "mps":
                    for name, module in self.model_loader.model.named_modules():
                        if hasattr(module, 'weight') and module.weight is not None:
                            if module.weight.dtype != torch.float16:
                                print(f"[DeepSeek-OCR] Converting {name}.weight from {module.weight.dtype} to float16")
                                module.weight.data = module.weight.data.to(torch.float16)
                        if hasattr(module, 'bias') and module.bias is not None:
                            if module.bias.dtype != torch.float16:
                                print(f"[DeepSeek-OCR] Converting {name}.bias from {module.bias.dtype} to float16")
                                module.bias.data = module.bias.data.to(torch.float16)

                result = self.model_loader.model.infer(
                    self.model_loader.tokenizer,
                    prompt=prompt,
                    image_file=tmp_path,
                    output_path=output_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False
                )

                text = self._extract_text(result)
                return (text,)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, dict):
            return result.get("text", result.get("result", str(result)))
        elif isinstance(result, (list, tuple)):
            return "\n".join(str(r) for r in result)
        return str(result)


class DeepSeekOCRFreeNode:
    """DeepSeek-OCR-2 Free OCR 节点"""

    def __init__(self):
        self.main_node = DeepSeekOCRNode()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "image_size": ("INT", {"default": 768, "min": 384, "max": 1536, "step": 64}),
                "crop_mode": ("BOOLEAN", {"default": True}),
                "model_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ocr_free"
    CATEGORY = "DeepSeek"

    def ocr_free(self, image: torch.Tensor, base_size: int, image_size: int,
                 crop_mode: bool, model_path: str) -> Tuple[str]:
        return self.main_node.ocr_process(
            image=image, prompt="<image>\nFree OCR. ",
            base_size=base_size, image_size=image_size, crop_mode=crop_mode,
            model_path=model_path, device="auto"
        )


NODE_CLASS_MAPPINGS = {
    "DeepSeekOCRNode": DeepSeekOCRNode,
    "DeepSeekOCRFreeNode": DeepSeekOCRFreeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekOCRNode": "DeepSeek-OCR-2",
    "DeepSeekOCRFreeNode": "DeepSeek-OCR-2 (Free Mode)",
}
