#!/usr/bin/env python3
"""
InkForgeAPI 测试客户端
用于测试 API 服务的功能
"""

import requests
import json
import base64
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np


class InkForgeAPIClient:
    """InkForgeAPI 客户端"""

    def __init__(self, base_url: str = "http://localhost:20802"):
        """
        初始化客户端

        Args:
            base_url: API 基础 URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "InkForgeAPIClient/1.0.0",
            "Accept": "application/json"
        })

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}

    def list_models(self) -> Dict[str, Any]:
        """列出所有模型"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}

    def predict_file(self, image_path: str, model_name: str = "chinese_handwriting", top_k: int = 5) -> Dict[str, Any]:
        """
        使用文件进行预测

        Args:
            image_path: 图像文件路径
            model_name: 模型名称
            top_k: 返回前K个预测结果

        Returns:
            预测结果
        """
        try:
            # 验证文件存在
            if not Path(image_path).exists():
                return {"success": False, "error": f"文件不存在: {image_path}"}

            # 准备文件上传
            files = {"file": (Path(image_path).name, open(image_path, "rb"), "image/jpeg")}
            params = {"model_name": model_name, "top_k": top_k}

            # 发送请求
            response = self.session.post(
                f"{self.base_url}/predict",
                files=files,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            # 关闭文件
            files["file"][1].close()

            return response.json()

        except requests.RequestException as e:
            return {"success": False, "error": f"请求失败: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"处理失败: {str(e)}"}

    def predict_base64(self, image_path: str, model_name: str = "chinese_handwriting", top_k: int = 5) -> Dict[str, Any]:
        """
        使用 base64 编码进行预测

        Args:
            image_path: 图像文件路径
            model_name: 模型名称
            top_k: 返回前K个预测结果

        Returns:
            预测结果
        """
        try:
            # 读取并编码图像
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # 准备请求数据
            data = {
                "image": image_b64,
                "model_name": model_name,
                "top_k": top_k
            }

            # 发送请求
            response = self.session.post(
                f"{self.base_url}/predict/base64",
                json=data,
                timeout=30
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            return {"success": False, "error": f"请求失败: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"处理失败: {str(e)}"}

    def create_test_image(self, output_path: str = "test_image.jpg", character: str = "一") -> bool:
        """
        创建测试图像

        Args:
            output_path: 输出文件路径
            character: 要绘制的字符

        Returns:
            是否成功
        """
        try:
            # 创建空白图像
            img_size = 200
            image = np.ones((img_size, img_size), dtype=np.uint8) * 255

            # 添加文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 5
            thickness = 10
            color = 0  # 黑色

            # 计算文字位置
            text_size = cv2.getTextSize(character, font, font_scale, thickness)[0]
            text_x = (img_size - text_size[0]) // 2
            text_y = (img_size + text_size[1]) // 2

            # 绘制文字
            cv2.putText(image, character, (text_x, text_y), font, font_scale, color, thickness)

            # 保存图像
            cv2.imwrite(output_path, image)
            print(f"测试图像已创建: {output_path}, 字符: {character}")
            return True

        except Exception as e:
            print(f"创建测试图像失败: {e}")
            return False

    def run_comprehensive_test(self, test_image_path: Optional[str] = None) -> bool:
        """
        运行综合测试

        Args:
            test_image_path: 测试图像路径，如果为 None 则创建测试图像

        Returns:
            测试是否全部通过
        """
        print("=" * 60)
        print("InkForgeAPI 综合测试")
        print("=" * 60)

        all_passed = True

        # 测试 1: 健康检查
        print("\n1. 健康检查测试...")
        health_result = self.health_check()
        if health_result.get("status") == "healthy":
            print(f"   ✓ 健康检查通过: {health_result}")
        else:
            print(f"   ✗ 健康检查失败: {health_result}")
            all_passed = False

        # 测试 2: 模型列表
        print("\n2. 模型列表测试...")
        models_result = self.list_models()
        if "models" in models_result:
            print(f"   ✓ 获取到 {len(models_result['models'])} 个模型")
            for model_name, model_info in models_result["models"].items():
                print(f"     - {model_name}: {model_info.get('description', '无描述')}")
        else:
            print(f"   ✗ 获取模型列表失败: {models_result}")
            all_passed = False

        # 创建测试图像
        if test_image_path is None:
            test_image_path = "test_api_image.jpg"
            if not self.create_test_image(test_image_path):
                print("   ✗ 创建测试图像失败")
                all_passed = False
                return all_passed

        # 测试 3: 文件上传预测
        print(f"\n3. 文件上传预测测试 (使用: {test_image_path})...")
        file_result = self.predict_file(test_image_path)
        if file_result.get("success"):
            predictions = file_result.get("predictions", [])
            print(f"   ✓ 文件上传预测成功")
            print(f"     模型: {file_result.get('model')}")
            print(f"     预测结果 (Top-{len(predictions)}):")
            for i, pred in enumerate(predictions, 1):
                print(f"       {i}. {pred.get('character')} (置信度: {pred.get('confidence'):.4f})")
        else:
            print(f"   ✗ 文件上传预测失败: {file_result.get('error', '未知错误')}")
            all_passed = False

        # 测试 4: Base64 预测
        print(f"\n4. Base64 预测测试 (使用: {test_image_path})...")
        base64_result = self.predict_base64(test_image_path)
        if base64_result.get("success"):
            predictions = base64_result.get("predictions", [])
            print(f"   ✓ Base64 预测成功")
            print(f"     模型: {base64_result.get('model')}")
            print(f"     预测结果 (Top-{len(predictions)}):")
            for i, pred in enumerate(predictions, 1):
                print(f"       {i}. {pred.get('character')} (置信度: {pred.get('confidence'):.4f})")
        else:
            print(f"   ✗ Base64 预测失败: {base64_result.get('error', '未知错误')}")
            all_passed = False

        # 测试 5: 性能测试
        print("\n5. 性能测试...")
        start_time = time.time()
        for i in range(3):
            result = self.predict_file(test_image_path)
            if not result.get("success"):
                print(f"   ✗ 第 {i+1} 次性能测试失败")
                all_passed = False
                break
        end_time = time.time()

        if all_passed:
            avg_time = (end_time - start_time) / 3
            print(f"   ✓ 性能测试通过，平均响应时间: {avg_time:.2f} 秒")

        # 清理测试图像
        if test_image_path.startswith("test_"):
            try:
                Path(test_image_path).unlink()
                print(f"\n清理测试图像: {test_image_path}")
            except:
                pass

        print("\n" + "=" * 60)
        if all_passed:
            print("所有测试通过！ ✓")
        else:
            print("部分测试失败！ ✗")
        print("=" * 60)

        return all_passed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="InkForgeAPI 测试客户端")
    parser.add_argument("--url", default="http://localhost:20802", help="API 基础 URL")
    parser.add_argument("--image", help="测试图像路径")
    parser.add_argument("--model", default="chinese_handwriting", help="模型名称")
    parser.add_argument("--top-k", type=int, default=5, help="返回前K个预测结果")
    parser.add_argument("--test", action="store_true", help="运行综合测试")
    parser.add_argument("--health", action="store_true", help="健康检查")
    parser.add_argument("--list-models", action="store_true", help="列出所有模型")
    parser.add_argument("--predict-file", help="使用文件进行预测")
    parser.add_argument("--predict-base64", help="使用 base64 进行预测")
    parser.add_argument("--create-test-image", help="创建测试图像")

    args = parser.parse_args()

    # 创建客户端
    client = InkForgeAPIClient(args.url)

    # 执行命令
    if args.health:
        result = client.health_check()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.list_models:
        result = client.list_models()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.predict_file:
        result = client.predict_file(args.predict_file, args.model, args.top_k)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.predict_base64:
        result = client.predict_base64(args.predict_base64, args.model, args.top_k)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.create_test_image:
        client.create_test_image(args.create_test_image)

    elif args.test:
        client.run_comprehensive_test(args.image)

    else:
        # 默认运行综合测试
        client.run_comprehensive_test(args.image)


if __name__ == "__main__":
    main()