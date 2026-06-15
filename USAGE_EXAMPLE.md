# 节点使用示例

## 基础连接方式

### 1. 单张图片反推提示词

```
┌─────────────────────┐
│  加载 Qwen 模型     │
│  (QwenModelLoader)  │
├─────────────────────┤
│  model_type: qwen2.5-vl-7b-instruct  │
│  device_map: auto    │
│  precision: float16  │
└─────────┬───────────┘
          │ model
          ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Load Image         │      │  反推提示词          │
│  (加载图片)          │      │  (QwenCaption)      │
├─────────────────────┤      ├─────────────────────┤
│                     │─ ─ ─▶│  image1             │
│  image: [你的图片]   │      │  task_type: auto    │
└─────────────────────┘      │  language: both     │
                             │  use_template: True │
                             └─────────┬───────────┘
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                    caption_combined  caption_chinese  caption_english
```

### 2. 多张图片综合分析

```
┌─────────────────────┐
│  加载 Qwen 模型     │
└─────────┬───────────┘
          │ model
          ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Load Image 1       │      │                     │
│  [主体图片]          │─ ─ ─▶│  image1             │
└─────────────────────┘      │                     │
                             │                     │
┌─────────────────────┐      │                     │
│  Load Image 2       │      │  反推提示词          │
│  [背景图片]          │─ ─ ─▶│  image2             │
└─────────────────────┘      │                     │
                             │  task_type: t2i     │
┌─────────────────────┐      │                     │
│  Load Image 3       │      │                     │
│  [参考风格]          │─ ─ ─▶│  image3             │
└─────────────────────┘      └─────────┬───────────┘
                                       │
                                       ▼
                                 生成综合提示词
```

### 3. 视频分析

```
┌─────────────────────┐
│  加载 Qwen 模型     │
└─────────┬───────────┘
          │ model
          ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Load Video         │      │                     │
│  (加载视频)          │─ ─ ─▶│  video              │
├─────────────────────┤      │                     │
│  video: [你的视频]   │      │  反推提示词          │
└─────────────────────┘      │  task_type: t2v     │
                             │  language: english  │
                             └─────────┬───────────┘
                                       │
                                       ▼
                                 生成视频描述
```

## 完整工作流示例

### 文生图 (T2I) 工作流

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 加载 Qwen   │    │ 加载参考图片  │    │ 反推提示词   │    │  文生图模型   │
│ 模型         │───▶│              │───▶│              │───▶│  (SD/Flux)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                          task_type: t2i
                                          use_template: True
```

### 图生视频 (I2V) 工作流

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 加载 Qwen   │    │ 加载图片     │    │ 反推提示词   │    │  图生视频    │
│ 模型         │───▶│              │───▶│              │───▶│  模型        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                          task_type: i2v
```

## 参数说明

### 加载模型节点

| 参数 | 选项 | 说明 |
|------|------|------|
| model_type | qwen2.5-vl-7b-instruct | 默认模型 |
| device_map | auto / cpu / cuda | 设备选择，auto 自动检测 |
| precision | float16 / bfloat16 / float32 | 精度，float16 最省显存 |

### 反推提示词节点

| 参数 | 选项 | 说明 |
|------|------|------|
| image1-5 | IMAGE | 最多 5 张图片输入 |
| video | VIDEO | 视频文件输入 |
| task_type | auto/t2i/t2v/i2i/i2v/r2i/r2v | 任务类型 |
| language | both/chinese/english | 输出语言 |
| use_template | True/False | 是否参考提示词模板 |
| max_tokens | 256-4096 | 最大生成长度 |

## 输出说明

| 输出 | 类型 | 说明 |
|------|------|------|
| caption_combined | STRING | 组合输出，包含中英文 |
| caption_chinese | STRING | 纯中文提示词 |
| caption_english | STRING | 纯英文提示词 |

## 提示词模板效果

启用 `use_template` 后，生成的提示词会包含：

- **光线描述**: Edge lighting, soft lighting, hard lighting
- **光源**: Daylight, Moonlight, Artificial lighting
- **镜头**: Medium shot, Close-up, Wide shot
- **构图**: Center composition, Balanced composition
- **色调**: Warm colors, Cool colors
- **主体细节**: 外貌、表情、姿态、服装等

示例输出：
```
【中文】
日光，中景镜头，居中构图，暖色调。一位年轻女性站在花园中，身穿白色连衣裙，
长发披肩，面带微笑。背景是盛开着各色鲜花的花丛，阳光透过树叶洒下斑驳光影。

【English】
Daylight, medium shot, center composition, warm colors. A young woman stands in
a garden, wearing a white dress with long hair flowing over her shoulders, smiling
gently. The background features blooming flowers in various colors, with dappled
sunlight filtering through the leaves.
```
