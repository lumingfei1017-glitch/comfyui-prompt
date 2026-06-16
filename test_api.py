"""
魔搭社区 API 节点测试脚本
用法: python test_api.py
"""

import json
import base64
import io
import requests
import urllib3
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== 配置 ==========
API_KEY = "ms-e96a3b60-9bfc-4645-80cc-33238bb6208b"
BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# 测试图片 URL（魔搭官方示例图）
TEST_IMAGE_URL = "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg"


NO_PROXY = {"http": None, "https": None}


def call_api(messages, max_tokens=1024):
    """使用 requests 直接调用 API（绕过代理直连）"""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120, verify=False, proxies=NO_PROXY)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def test_api_connection():
    """测试 API 连接"""
    print("=" * 50)
    print("测试 1: API 连接")
    print("=" * 50)

    try:
        resp = requests.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
            verify=False,
            proxies=NO_PROXY
        )
        print(f"✓ API 连接成功，状态码: {resp.status_code}")
        return True
    except Exception as e:
        print(f"✗ API 连接失败: {e}")
        return False


def test_text_only():
    """测试纯文本调用"""
    print("\n" + "=" * 50)
    print("测试 2: 纯文本调用")
    print("=" * 50)

    try:
        result = call_api([{"role": "user", "content": "你好，请用一句话介绍自己"}], max_tokens=100)
        print(f"✓ 纯文本调用成功")
        print(f"  回复: {result[:200]}")
        return True
    except Exception as e:
        print(f"✗ 纯文本调用失败: {e}")
        return False


def test_image_url():
    """测试图片 URL 调用"""
    print("\n" + "=" * 50)
    print("测试 3: 图片 URL 调用")
    print("=" * 50)

    try:
        result = call_api([{
            "role": "user",
            "content": [
                {"type": "text", "text": "请用中文描述这张图片的内容"},
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}}
            ]
        }], max_tokens=300)
        print(f"✓ 图片 URL 调用成功")
        print(f"  回复: {result[:500]}")
        return True
    except Exception as e:
        print(f"✗ 图片 URL 调用失败: {e}")
        return False


def test_image_base64():
    """测试 base64 图片调用（模拟 ComfyUI 节点的实际用法）"""
    print("\n" + "=" * 50)
    print("测试 4: Base64 图片调用")
    print("=" * 50)

    try:
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{b64_str}"

        result = call_api([{
            "role": "user",
            "content": [
                {"type": "text", "text": "这是什么颜色的图片？"},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        }], max_tokens=100)
        print(f"✓ Base64 图片调用成功")
        print(f"  回复: {result[:300]}")
        return True
    except Exception as e:
        print(f"✗ Base64 图片调用失败: {e}")
        return False


def test_prompt_generation():
    """测试完整的提示词生成（模拟节点实际流程）"""
    print("\n" + "=" * 50)
    print("测试 5: 提示词生成（完整流程）")
    print("=" * 50)

    prompt = """请分析提供的1张图片内容，生成用于AI文生图的详细提示词。

请按照以下JSON格式返回：
{
    "中文提示词": "详细的中文描述",
    "英文提示词": "Detailed English description"
}

要求：
1. 描述要详细，包含主体、细节、风格、构图、光线等元素
2. 英文提示词要适合AI模型使用"""

    try:
        result = call_api([{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
                {"type": "text", "text": prompt}
            ]
        }], max_tokens=1024)
        print(f"✓ 提示词生成成功")

        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = result[json_start:json_end]
            json_data = json.loads(json_str)
            cn = json_data.get('中文提示词', '')
            en = json_data.get('英文提示词', '')
            print(f"  中文: {cn[:200]}")
            print(f"  英文: {en[:200]}")
        else:
            print(f"  原始回复: {result[:500]}")

        return True
    except Exception as e:
        print(f"✗ 提示词生成失败: {e}")
        return False


def main():
    print("魔搭社区 API 节点测试")
    print("=" * 50)

    if not test_api_connection():
        print("\n测试终止: API 连接失败")
        return

    results = []
    results.append(("纯文本", test_text_only()))
    results.append(("图片URL", test_image_url()))
    results.append(("Base64图片", test_image_base64()))
    results.append(("提示词生成", test_prompt_generation()))

    print("\n" + "=" * 50)
    print("测试汇总")
    print("=" * 50)
    for name, ok in results:
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name}: {status}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n总计: {passed}/{len(results)} 通过")


if __name__ == "__main__":
    main()
