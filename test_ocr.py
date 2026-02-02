#!/usr/bin/env python3
"""
测试 DeepSeek-OCR-2 节点
使用本地已下载的模型进行测试
"""
import os
import sys
import tempfile
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 设置本地模型路径
LOCAL_MODEL_PATH = "/Users/xiohu/work/ai-tools/models/DeepSeek-OCR-2"

# 导入节点
from nodes import DeepSeekOCRNode, DeepSeekOCRFreeNode


def create_test_image():
    """创建一个测试图像"""
    img = Image.new('RGB', (800, 200), color='white')

    draw = ImageDraw.Draw(img)
    text = "Hello DeepSeek OCR! This is a test."
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()

    draw.text((50, 80), text, fill='black', font=font)
    return img


def image_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """将 PIL Image 转换为 ComfyUI 格式的 tensor"""
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)
    # ComfyUI 格式: [B, H, W, C]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def test_ocr_node():
    """测试 DeepSeek OCR 节点"""
    print("=" * 60)
    print("DeepSeek-OCR-2 节点测试")
    print("=" * 60)

    # 检查本地模型
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"错误: 本地模型路径不存在: {LOCAL_MODEL_PATH}")
        return False

    print(f"✓ 本地模型路径: {LOCAL_MODEL_PATH}")

    # 创建测试图像
    print("\n创建测试图像...")
    test_image = create_test_image()
    image_tensor = image_to_tensor(test_image)
    print(f"✓ 测试图像 tensor 形状: {image_tensor.shape}")

    # 创建节点实例
    print("\n创建节点...")
    node = DeepSeekOCRNode()
    print("✓ 节点已创建")

    try:
        # 测试 Free OCR 模式
        print("\n" + "-" * 60)
        print("测试模式: Free OCR")
        print("-" * 60)

        result = node.ocr_process(
            image=image_tensor,
            prompt="<image>\nFree OCR. ",
            base_size=1024,
            image_size=768,
            crop_mode=True,
            model_path=LOCAL_MODEL_PATH,
            device="auto"
        )

        print("\n识别结果:")
        print("-" * 60)
        print(result[0])
        print("-" * 60)

        print("\n✓ 测试成功!")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_free_node():
    """测试 Free OCR 节点"""
    print("\n" + "=" * 60)
    print("DeepSeek-OCR-2 (Free Mode) 节点测试")
    print("=" * 60)

    # 创建测试图像
    print("\n创建测试图像...")
    test_image = create_test_image()
    image_tensor = image_to_tensor(test_image)

    # 创建节点实例
    node = DeepSeekOCRFreeNode()

    try:
        print("\n" + "-" * 60)
        print("测试模式: Free OCR Node")
        print("-" * 60)

        result = node.ocr_free(
            image=image_tensor,
            base_size=1024,
            image_size=768,
            crop_mode=True,
            model_path=LOCAL_MODEL_PATH,
        )

        print("\n识别结果:")
        print("-" * 60)
        print(result[0])
        print("-" * 60)

        print("\n✓ 测试成功!")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_ocr_node()
    success2 = test_free_node()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"DeepSeek-OCR-2 节点: {'✓ 通过' if success1 else '✗ 失败'}")
    print(f"DeepSeek-OCR-2 (Free Mode) 节点: {'✓ 通过' if success2 else '✗ 失败'}")

    sys.exit(0 if (success1 and success2) else 1)
