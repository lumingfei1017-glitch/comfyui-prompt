# Qwen-CLIP ComfyUI Plugin

基于 Qwen2.5-VL-7B-Instruct 模型的图片/视频提示词反推插件。

## 功能特点

- 🤖 基于 Qwen2.5-VL-7B-Instruct 视觉语言模型
- 🖼️ 支持最多 5 张图片同时输入分析
- 🎬 支持视频输入（自动提取关键帧）
- 📝 参考提示词模板生成专业提示词
- 🌍 支持中英文双语输出
- 🔄 自动下载和管理模型

## 节点说明

### 1. 加载 Qwen 模型 (QwenModelLoaderNode)

加载 Qwen2.5-VL 视觉语言模型，供下游节点使用。

**输入参数：**
- `model_type` - 模型类型（默认: qwen2.5-vl-7b-instruct）
- `device_map` - 设备映射（auto/cpu/cuda）
- `precision` - 模型精度（float16/bfloat16/float32）

**输出：**
- `model` - 模型对象，传递给反推提示词节点

### 2. 反推提示词 (QwenCaptionGeneratorNode)

使用加载的模型分析图片/视频，生成 AI 生成任务的提示词。

**输入参数：**
- `model` - 从加载模型节点接收的模型对象
- `image1` ~ `image5` - 5 个图片输入接口（可选）
- `video` - 视频输入接口（可选）
- `task_type` - 任务类型（auto/t2i/t2v/i2i/i2v/r2i/r2v）
- `language` - 输出语言（both/chinese/english）
- `use_template` - 是否使用提示词模板
- `max_tokens` - 最大生成 token 数

**输出：**
- `caption_combined` - 组合提示词（中英文）
- `caption_chinese` - 中文提示词
- `caption_english` - 英文提示词

## 安装

1. 进入 ComfyUI 的 `custom_nodes` 目录
2. 克隆此仓库：
   ```bash
   git clone https://github.com/your-username/qwen-clip.git
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 基础用法：单张图片反推

```
[加载 Qwen 模型] → model → [反推提示词] → caption
                         ↑
                    [加载图片] → image1
```

### 多图片分析

```
[加载 Qwen 模型] → model → [反推提示词] → caption
                         ↑
                    [图片1] → image1
                    [图片2] → image2
                    [图片3] → image3
```

### 视频分析

```
[加载 Qwen 模型] → model → [反推提示词] → caption
                         ↑
                    [视频] → video
```

## 任务类型说明

| 任务类型 | 说明 | 适用场景 |
|---------|------|---------|
| auto | 自动检测 | 根据输入自动判断 |
| t2i | 文生图 | 生成图片描述 |
| t2v | 文生视频 | 生成视频描述 |
| i2i | 图像编辑 | 图片修改描述 |
| i2v | 图生视频 | 图片转视频描述 |
| r2i | 主体参考生图 | 参考图片生成 |
| r2v | 主体参考生视频 | 参考视频生成 |

## 提示词模板

插件会自动读取 `提示词模板.txt` 文件，其中包含：

- 电影美学设定规则
- 各任务类型的系统提示词
- 输出格式要求

启用模板后，生成的提示词会遵循模板中的规则，包含：
- 光线/光源描述
- 镜头角度/构图
- 色调/氛围
- 主体细节描述

## 模型管理

插件会自动在 `ComfyUI/models/clip` 目录下管理模型：

- 首次使用时自动下载所需模型
- 自动检测已存在的模型
- 支持 Hugging Face 镜像下载

## 注意事项

1. 模型较大（约 15GB），首次加载需要较长时间
2. 建议使用 16GB 以上显存的 GPU
3. 多图片输入时，插件会综合分析所有图片生成统一提示词
4. 视频输入会自动提取 5 个关键帧进行分析
5. 首次运行需要下载模型，建议有足够的磁盘空间