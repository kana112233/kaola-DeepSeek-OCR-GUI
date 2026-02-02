#!/usr/bin/env python3
"""
测试 GUI 程序的 OCR 功能
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from deepseek_ocr_gui import DeepSeekOCR


def test_ocr():
    """测试 OCR 功能"""
    print("=" * 60)
    print("DeepSeek-OCR-2 GUI 核心功能测试")
    print("=" * 60)

    # 创建 OCR 实例
    ocr = DeepSeekOCR()

    # 测试图像路径
    test_image = "test_image.jpg"

    if not os.path.exists(test_image):
        print(f"错误: 测试图像不存在: {test_image}")
        return False

    print(f"\n测试图像: {test_image}")

    # 加载模型
    print("\n1. 加载模型...")
    try:
        ocr.load(lambda msg: print(f"   {msg}"))
        print("   ✓ 模型加载成功")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        return False

    # 测试 Free OCR 模式
    print("\n2. 测试 Free OCR 模式...")
    try:
        result_free = ocr.ocr(
            test_image,
            mode="free",
            progress_callback=lambda msg: print(f"   {msg}")
        )
        print(f"   识别结果: {result_free[:50]}...")
        print("   ✓ Free OCR 模式测试通过")
    except Exception as e:
        print(f"   ✗ Free OCR 模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 Markdown 模式
    print("\n3. 测试 Markdown 模式...")
    try:
        result_md = ocr.ocr(
            test_image,
            mode="markdown",
            progress_callback=lambda msg: print(f"   {msg}")
        )
        print(f"   识别结果: {result_md[:50]}...")
        print("   ✓ Markdown 模式测试通过")
    except Exception as e:
        print(f"   ✗ Markdown 模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_ocr()
    sys.exit(0 if success else 1)
