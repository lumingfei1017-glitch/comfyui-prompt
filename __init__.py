"""
Qwen-CLIP ComfyUI Plugin
图片/视频提示词反推插件

包含两个节点：
1. QwenModelLoaderNode - 加载 Qwen2.5-VL 模型
2. QwenCaptionGeneratorNode - 反推提示词（支持多图片和视频输入）
"""

from .qwen_clip_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 导出节点映射
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Web 目录（如果有前端组件）
WEB_DIRECTORY = None
