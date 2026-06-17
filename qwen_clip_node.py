import os
import json
import base64
import io
import numpy as np
from PIL import Image
import folder_paths
import requests as req


# ========== 模板定义 ==========
# 根据提示词模板.txt 提取的各任务类型系统提示词

SYSTEM_PROMPTS = {
    "t2i": """You are a helpful assistant specialized in text-to-image generation.
你是一位电影导演，旨在为用户输入的原始prompt添加电影元素，改写为优质（英文）Prompt，使其完整、具有表现力。注意，输出必须是英文！
本任务为文生图 (text-to-image)，请按下列电影美学规则改写为静态图像 prompt。图像里没有时间序列，不要描述运动/摄像机运动/动作过程，只描写场景与主体的静态状态。其余电影美学（光源/光线强度/色调/镜头大小/拍摄角度/构图）按下文规则保留。

任务要求：
1. 对于用户输入的prompt，在不改变prompt的原意（如主体、动作）前提下，从下列电影美学设定中选择不超过4种合适的时间、光源、光线强度、光线角度、对比度、饱和度、色调、拍摄角度、镜头大小、构图的电影设定细节，将这些内容添加到prompt中，让画面变得更美。可以任选，不必每项都有：
  时间：["Day time", "Night time", "Dawn time", "Sunrise time"]，如果prompt没有特别说明则选 Day time
  光源：["Daylight", "Artificial lighting", "Moonlight", "Practical lighting", "Firelight", "Fluorescent lighting", "Overcast lighting", "Sunny lighting"]，根据室内室外及prompt内容选定义光源
  光线强度：["Soft lighting", "Hard lighting"]
  色调：["Warm colors", "Cool colors", "Mixed colors"]
  光线角度：["Top lighting", "Side lighting", "Underlighting", "Edge lighting"]
  镜头尺寸：["Medium shot", "Medium close-up shot", "Wide shot", "Medium wide shot", "Close-up shot", "Extreme close-up shot", "Extreme wide shot"]，若无特殊要求默认Medium shot或Wide shot
  拍摄角度：["Over-the-shoulder shot", "Low angle shot", "High angle shot", "Dutch angle shot", "Aerial shot", "Overhead shot"]
  构图：["Center composition", "Balanced composition", "Right-heavy composition", "Left-heavy composition", "Symmetrical composition", "Short-side composition"]，若无特殊要求默认Center composition
2. 完善用户描述中出现的主体特征（外貌、表情、数量、种族、姿态等），确保不要添加原始prompt中不存在的主体，增加背景元素的细节
3. 不要输出关于氛围、感觉等文学描写
4. 不要描述运动/摄像机运动/动作过程，只描写主体和背景的静态状态、姿态、表情、构图等
5. 若原始prompt中没有风格，则不添加风格描述；若有风格描述，则将风格描述放于首位；若为2D插画等与现实电影相悖的风格，则不要添加关于电影美学的描写
6. 若prompt出现天空的描述，则改为湛蓝色的天空相关描述，避免曝光
7. 输出必须是全英文，改写后的prompt字数控制在60-200字左右，不要输出类似"改写后prompt:"这样的前缀""",

    "t2v": """You are a helpful assistant specialized in text-to-video generation.
你是一位电影导演，旨在为用户输入的原始prompt添加电影元素，改写为优质（英文）Prompt，使其完整、具有表现力。注意，输出必须是英文！

任务要求：
1. 对于用户输入的prompt，在不改变prompt的原意（如主体、动作）前提下，从下列电影美学设定中选择不超过4种合适的时间、光源、光线强度、光线角度、对比度、饱和度、色调、拍摄角度、镜头大小、构图的电影设定细节，将这些内容添加到prompt中，让画面变得更美。可以任选，不必每项都有：
  时间：["Day time", "Night time", "Dawn time", "Sunrise time"]，如果prompt没有特别说明则选 Day time
  光源：["Daylight", "Artificial lighting", "Moonlight", "Practical lighting", "Firelight", "Fluorescent lighting", "Overcast lighting", "Sunny lighting"]，根据室内室外及prompt内容选定义光源
  光线强度：["Soft lighting", "Hard lighting"]
  色调：["Warm colors", "Cool colors", "Mixed colors"]
  光线角度：["Top lighting", "Side lighting", "Underlighting", "Edge lighting"]
  镜头尺寸：["Medium shot", "Medium close-up shot", "Wide shot", "Medium wide shot", "Close-up shot", "Extreme close-up shot", "Extreme wide shot"]，若无特殊要求默认Medium shot或Wide shot
  拍摄角度：["Over-the-shoulder shot", "Low angle shot", "High angle shot", "Dutch angle shot", "Aerial shot", "Overhead shot"]
  构图：["Center composition", "Balanced composition", "Right-heavy composition", "Left-heavy composition", "Symmetrical composition", "Short-side composition"]，若无特殊要求默认Center composition
2. 完善用户描述中出现的主体特征（外貌、表情、数量、种族、姿态等），确保不要添加原始prompt中不存在的主体，增加背景元素的细节
3. 不要输出关于氛围、感觉等文学描写
4. 对于prompt中的动作，详细描述运动的发生过程，若没有动作则添加动作描述（摇晃身体、跳舞等），对背景元素也可添加适当运动（如云彩飘动、风吹树叶等）
5. 若原始prompt中没有风格，则不添加风格描述；若有风格描述，则将风格描述放于首位；若为2D插画等与现实电影相悖的风格，则不要添加关于电影美学的描写
6. 若prompt出现天空的描述，则改为湛蓝色的天空相关描述，避免曝光
7. 输出必须是全英文，改写后的prompt字数控制在60-200字左右，不要输出类似"改写后prompt:"这样的前缀""",

    "i2i": """You are a helpful assistant specialized in image editing.
Task: Image Editing
# ROLE
You are an expert Image-to-Image (I2I) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source image to generate a detailed I2I editing prompt in English.

# CORE GENERATION RULE
Unless specified otherwise by the task type, your generated prompt MUST strictly follow this two-part structure:
1. Modifications: Specifically describe what needs to be changed.
2. Preservations: Explicitly describe the key visual elements, background, or subjects that MUST remain unchanged.
3. Concretization: If the user's instruction contains vague references, you MUST replace them with specific, well-known, named instances.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not include any explanations, greetings, or the category name.
Do not imagine things that do not appear in the image.""",

    "r2i": """You are an expert at writing subject-driven image generation prompts. I'm providing you with reference image(s) of the subject(s) that will appear in the generated image.

Your task is to analyze the reference image(s) and generate a detailed prompt for subject-driven image generation:

**Requirements:**
- Describe the subject(s) from the reference image(s) with detailed appearance (hair, clothing, accessories, expression, etc.)
- Describe the scene/environment in detail (background, lighting, objects, atmosphere)
- Describe the composition, framing, and visual emphasis
- The appearance description of each subject must be based on what you actually see in the reference image(s). Do NOT hallucinate details not visible in the images
- The output must be entirely in English
- Output ONLY the final enhanced English prompt. No extra text.""",

    "r2v": """You are an expert at writing subject-driven video generation prompts. I'm providing you with reference image(s) of the subject(s) that will appear in the video.

Your task is to analyze the reference image(s) and generate a detailed prompt for subject-driven video generation:

**Requirements:**
- Describe the subject(s) from the reference image(s) with detailed appearance (hair, clothing, accessories, expression, etc.)
- Describe the scene/environment in detail (background, lighting, objects, atmosphere)
- Describe the motion and actions in a step-by-step temporal sequence (at the start..., then..., after that...)
- The motion should remain natural and realistic
- The appearance description of each subject must be based on what you actually see in the reference image(s). Do NOT hallucinate details not visible in the images
- The output must be entirely in English
- Output ONLY the final enhanced English prompt. No extra text.""",

    "i2v": """You are a helpful assistant specialized in image-to-video generation.
Task: Image-to-Video Generation

I'm providing reference image(s) used as input frames. Your task is to analyze the image(s) and generate a detailed English prompt for video generation.

**Requirements:**
- If 1 image: Generate a video description prompt describing actions, camera movements, and scene, following T2V cinematic style
- If 2 images: Return "Generate a video based on the first and last frames." + video description
- If 3 images: Return "Generate a video based on the first, middle, and last frames." + video description
- Output ONLY the final English prompt. No extra explanations.""",

    "v2v": """You are an expert Video-to-Video (V2V) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source video frames to generate a detailed V2V editing prompt in English.

# CORE GENERATION RULE
Your generated prompt MUST strictly follow this two-part structure:
1. Modifications: Specifically describe what needs to be changed. Include details like physical appearance, spatial location, lighting, and motion tracking.
2. Preservations: Explicitly describe the key visual elements, background, or subjects that MUST remain unchanged.
3. Concretization: If the user's instruction contains vague references to characters, objects, outfits, or styles, you MUST replace them with specific, well-known, named instances.

Note that you don't need to explicitly write "Modifications: xx. Preservations: xx.". Just describe it naturally.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not include any explanations, greetings, or the category name.
Do not imagine things that do not appear in the video.""",

    "vi2v": """You are an expert at video editing with reference images. Your task is to analyze the source video frames and reference image(s), then generate a detailed English prompt for reference-guided video editing.

**Task types may include:**
- Propagation: edit the video following the first frame
- Reference insertion: integrate the object from the reference image into the video
- Reference replacement: replace an element in the video with the element from the reference image

**Requirements:**
- Analyze the provided images to determine the appropriate task type
- Generate a precise English editing prompt
- If it's a propagation task, output: "edit the video following the first frame."
- If it's an insertion/replacement task, describe what to integrate/replace based on what you see
- Output ONLY the final English prompt. No extra explanations.""",

    "rv2v": """You are an expert at writing prompts for reference-image-guided video editing. I'm providing you with:
1. The first images are uniformly sampled frames from the source video that will be edited (in temporal order: frame0, frame1, frame2).
2. The next images are reference image(s) that should guide the editing.
3. An original editing instruction may be provided.

The reference image(s) may serve different roles depending on the editing task — for example, providing the target object/person for a replacement or addition, indicating a target visual style, demonstrating a target motion or pose, or guiding other attribute-level edits.

Your task: Rewrite and enhance the original editing instruction into a detailed, precise English prompt for a reference-image-guided video editing model. The output is a single paragraph in the format: editing instruction + detailed description of the target edited video, concatenated together.

Follow these rules strictly:

1. Output format: an editing instruction sentence followed by a detailed description of what the target video should look like, written as one continuous paragraph.
2. Match the edit type: use the verb that matches the actual intent — "Replace...", "Remove...", "Add...", "Restyle... in the style of...", "Transfer the motion/pose of... to...", "Change the ... of ...", etc.
3. Add ≠ Replace: for addition tasks, write them as additions, never as replacements.
4. Describe the target video directly: do not use phrases like "after editing..." or "in the edited video...".
5. Faithful reference appearance: the appearance, clothing, color, material in the prompt must match what is actually visible in the reference image. Do not hallucinate details.
6. Screen-perspective left/right: all left/right directions must be from the camera/screen perspective.
7. Preserve unchanged elements explicitly: state which aspects remain unchanged.
8. No parentheses: do NOT use parentheses "()" anywhere in the output.
9. English only: the output must be entirely in English.

Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten editing prompt as one string. No extra text.""",

    "ads2v": """You are an expert at writing prompts for video advertising insertion. I'm providing you with uniformly sampled frames from the source video for context.

Your task is to generate a concise English advertising insertion instruction based on what you see in the video frames.

**Requirements:**
- Analyze the video frames to understand the scene
- Generate a simple, one-sentence English instruction for ad placement
- Example format: "Add Starbucks Latte wallpaper on the second floor across the street"
- Output ONLY the final English prompt. No extra explanations.""",
}

# auto 类型不使用系统提示词，直接分析图片
AUTO_SYSTEM_PROMPT = """You are a helpful assistant. Analyze the provided image(s) and generate a detailed English prompt suitable for AI image/video generation.

Requirements:
1. Describe the main subject, details, style, composition, lighting, and atmosphere
2. The prompt should be detailed and suitable for AI generation models
3. If multiple images, analyze them comprehensively and generate a unified prompt
4. Output ONLY the English prompt. No extra explanations."""


class ModelScopeAPILoaderNode:
    """通过魔搭社区 API 加载模型（无需本地 GPU）"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "魔搭社区 API Token，前往 https://modelscope.cn 获取"
                }),
            },
            "optional": {
                "model": (["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"], {
                    "default": "Qwen/Qwen3-VL-8B-Instruct"
                }),
                "base_url": ("STRING", {
                    "default": "https://api-inference.modelscope.cn/v1"
                }),
            }
        }

    RETURN_TYPES = ("MODELSCOPE_API",)
    RETURN_NAMES = ("api_config",)
    FUNCTION = "load_api"
    CATEGORY = "QwenCLIP"

    def load_api(self, api_key, model="Qwen/Qwen3-VL-8B-Instruct",
                 base_url="https://api-inference.modelscope.cn/v1"):
        """返回 API 配置"""
        if not api_key or api_key.strip() == "":
            raise Exception("请填写魔搭社区 API Token，前往 https://modelscope.cn 控制台获取")

        api_config = {
            "base_url": base_url,
            "api_key": api_key.strip(),
            "model": model
        }

        print(f"魔搭 API 配置完成: model={model}, base_url={base_url}")
        return (api_config,)


class ModelScopeAPICaptionNode:
    """通过魔搭社区 API 反推提示词（无需本地 GPU）"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_config": ("MODELSCOPE_API",),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "video_batch": ("IMAGE", {
                    "tooltip": "视频拆帧后的图像批次（如使用VHS的Video Loader节点）。将自动取首帧和尾帧进行分析"
                }),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "补充文本描述（如原始提示词、编辑指令等），将与图片一起发送给模型分析"
                }),
                "task_type": (["auto", "t2i", "t2v", "i2i", "i2v", "r2i", "r2v", "v2v", "vi2v", "rv2v", "ads2v"], {
                    "default": "auto"
                }),
                "language": (["both", "chinese", "english"], {
                    "default": "both"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("template_output", "caption_chinese", "caption_english")
    FUNCTION = "generate_caption"
    CATEGORY = "QwenCLIP"

    def image_to_base64_url(self, image_tensor):
        """将 ComfyUI 图像张量转换为 base64 data URL（压缩以减小请求体积）"""
        image_np = image_tensor[0].cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # 缩放到最大 1024 边长，减小 base64 体积
        max_size = 1024
        w, h = pil_image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        # 用 JPEG 压缩，比 PNG 小很多
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def extract_first_last_frames(self, video_batch):
        """从图像批次中提取首帧和尾帧"""
        if video_batch is None or len(video_batch) == 0:
            return None, None

        batch_size = len(video_batch)
        print(f"[QwenCLIP] 视频批次大小: {batch_size} 帧")

        # 取第一帧
        first_frame = video_batch[0:1]  # 保持 batch 维度

        # 取最后一帧
        last_frame = video_batch[-1:]  # 保持 batch 维度

        return first_frame, last_frame

    def build_messages(self, images, task_type, language, video_frames=None, user_text_input=""):
        """根据任务类型和图片构建 API 消息"""
        # 选择系统提示词
        if task_type == "auto":
            system_prompt = AUTO_SYSTEM_PROMPT
        else:
            system_prompt = SYSTEM_PROMPTS.get(task_type, AUTO_SYSTEM_PROMPT)

        # 构建用户消息内容
        content = []

        # 添加视频帧（如果有）
        if video_frames:
            for frame_tensor in video_frames:
                if frame_tensor is not None:
                    img_url = self.image_to_base64_url(frame_tensor)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_url}
                    })

        # 添加普通图片输入
        for img in images:
            img_url = self.image_to_base64_url(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })

        # 根据语言要求调整输出指令
        if language == "chinese":
            lang_instruction = "\n\n请用中文输出最终的提示词。"
        elif language == "english":
            lang_instruction = "\n\nOutput in English only."
        else:
            lang_instruction = "\n\n请按照以下JSON格式返回：\n{\"中文提示词\": \"详细的中文描述\", \"英文提示词\": \"Detailed English description\"}"

        # 计算总图片数量
        total_images = len(content)  # content 中只有图片，还没有添加文本
        has_video = video_frames and len(video_frames) > 0

        # 构建用户文本
        if user_text_input and user_text_input.strip():
            # 有用户输入的文本
            if task_type != "auto":
                if has_video:
                    user_text = f"参考以下文本描述：\n{user_text_input.strip()}\n\n请结合提供的{total_images}张图片（含视频首尾帧），根据任务类型「{task_type}」的要求，生成对应的提示词。{lang_instruction}"
                else:
                    user_text = f"参考以下文本描述：\n{user_text_input.strip()}\n\n请结合提供的{total_images}张图片，根据任务类型「{task_type}」的要求，生成对应的提示词。{lang_instruction}"
            else:
                if has_video:
                    user_text = f"参考以下文本描述：\n{user_text_input.strip()}\n\n请结合提供的{total_images}张图片（含视频首尾帧）内容，生成详细的AI生成提示词。{lang_instruction}"
                else:
                    user_text = f"参考以下文本描述：\n{user_text_input.strip()}\n\n请结合提供的{total_images}张图片内容，生成详细的AI生成提示词。{lang_instruction}"
        else:
            # 没有用户输入的文本
            if task_type != "auto":
                if has_video:
                    user_text = f"请分析提供的{total_images}张图片（含视频首尾帧），根据任务类型「{task_type}」的要求，生成对应的提示词。{lang_instruction}"
                else:
                    user_text = f"请分析提供的{total_images}张图片，根据任务类型「{task_type}」的要求，生成对应的提示词。{lang_instruction}"
            else:
                if has_video:
                    user_text = f"请分析提供的{total_images}张图片（含视频首尾帧）内容，生成详细的AI生成提示词。{lang_instruction}"
                else:
                    user_text = f"请分析提供的{total_images}张图片内容，生成详细的AI生成提示词。{lang_instruction}"

        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        return messages

    def call_api(self, api_config, messages):
        """调用魔搭 API"""
        url = f"{api_config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": api_config["model"],
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
        }

        print(f"正在调用魔搭 API: {api_config['model']}...")

        # 绕过代理直连（平台代理会丢弃大请求）
        no_proxy = {"http": None, "https": None}
        resp = req.post(url, headers=headers, json=payload, timeout=120, verify=False, proxies=no_proxy)
        resp.raise_for_status()

        resp_data = resp.json()
        result = resp_data["choices"][0]["message"]["content"]
        print(f"API 返回完成，长度: {len(result)}")
        return result

    def parse_response(self, result, language):
        """解析 API 响应"""
        template_output = result.strip()
        chinese_caption = ""
        english_caption = ""

        if language == "chinese":
            chinese_caption = template_output
            english_caption = ""
        elif language == "english":
            english_caption = template_output
            chinese_caption = ""
        else:
            # both 模式：尝试解析 JSON
            try:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    json_data = json.loads(json_str)
                    chinese_caption = json_data.get('中文提示词', '')
                    english_caption = json_data.get('英文提示词', '')
                else:
                    chinese_caption = result
                    english_caption = result
            except json.JSONDecodeError:
                chinese_caption = result
                english_caption = result

        return template_output, chinese_caption, english_caption

    def generate_caption(self, api_config, image1=None, image2=None, image3=None,
                         image4=None, image5=None, video_batch=None, text="", task_type="auto", language="both"):
        """通过魔搭 API 生成图像描述"""
        try:
            # 收集所有输入的图片
            images = [img for img in [image1, image2, image3, image4, image5] if img is not None]

            # 处理视频批次：提取首尾帧
            video_frames = None
            if video_batch is not None and len(video_batch) > 0:
                first_frame, last_frame = self.extract_first_last_frames(video_batch)
                if first_frame is not None and last_frame is not None:
                    video_frames = [first_frame, last_frame]
                    print(f"[QwenCLIP] 已从视频批次提取首尾帧")

            # 检查是否有有效输入（图片、视频或文本）
            if not images and not video_frames and not text.strip():
                raise Exception("请至少提供一张图片、视频批次或输入文本描述")

            # 如果只有视频没有图片，自动设置任务类型为视频相关
            if video_frames and not images and task_type == "auto":
                task_type = "t2v"
                print(f"[QwenCLIP] 检测到视频输入，自动切换任务类型为: t2v")

            # 构建消息
            messages = self.build_messages(images, task_type, language, video_frames, text)

            # 调用 API
            result = self.call_api(api_config, messages)

            # 解析响应
            template_output, chinese_caption, english_caption = self.parse_response(result, language)

            return (template_output, chinese_caption, english_caption)

        except Exception as e:
            raise Exception(f"API 调用失败: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ModelScopeAPILoaderNode": ModelScopeAPILoaderNode,
    "ModelScopeAPICaptionNode": ModelScopeAPICaptionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeAPILoaderNode": "魔搭 API 配置",
    "ModelScopeAPICaptionNode": "反推提示词 (魔搭 API)",
}
