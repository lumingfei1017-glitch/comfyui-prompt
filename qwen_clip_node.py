import os
import json
import torch
import numpy as np
from PIL import Image
import folder_paths
from .model_manager import ModelManager

# Qwen2.5-VL 使用 AutoProcessor 而非 AutoTokenizer
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenModelLoaderNode:
    """加载 Qwen2.5-VL 模型节点"""

    def __init__(self):
        self.model_manager = ModelManager()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["qwen2.5-vl-7b-instruct"], {
                    "default": "qwen2.5-vl-7b-instruct"
                }),
            },
            "optional": {
                "device_map": (["auto", "cpu", "cuda"], {
                    "default": "auto"
                }),
                "precision": (["float16", "bfloat16", "float32"], {
                    "default": "float16"
                }),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "QwenCLIP"

    def load_model(self, model_type, device_map="auto", precision="float16"):
        """加载模型并返回模型对象"""
        try:
            # 获取模型路径
            model_path = self.model_manager.get_model_path(model_type)
            if not model_path:
                print(f"模型未找到，开始下载: {model_type}")
                model_path = self.model_manager.download_model(model_type)

            print(f"正在加载模型: {model_path}")

            # 导入模型类
            from transformers import Qwen2_5_VLForConditionalGeneration

            # 加载处理器（Qwen2.5-VL 使用 Processor 而非 Tokenizer）
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # 创建卸载目录
            offload_folder = os.path.join(os.path.dirname(model_path), "offload")
            os.makedirs(offload_folder, exist_ok=True)

            # 根据精度设置dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            torch_dtype = dtype_map.get(precision, torch.float16)

            # 尝试加载模型
            model = None
            try:
                print(f"尝试以 Qwen2_5_VLForConditionalGeneration 加载...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
                print("模型已加载（Qwen2_5_VLForConditionalGeneration）")
            except Exception as e:
                # 尝试从 Hub 加载
                print(f"本地加载失败，尝试从 Hugging Face Hub 加载: {e}")
                model_name = os.path.basename(model_path)
                hub_model_id = f"Qwen/{model_name}"

                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    hub_model_id,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
                print("模型已从 Hugging Face Hub 加载")

            print("模型加载完成")

            # 返回模型和处理器的组合
            model_info = {
                "model": model,
                "processor": processor,
                "model_type": model_type,
                "model_path": model_path
            }

            return (model_info,)

        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")


class QwenCaptionGeneratorNode:
    """反推提示词节点 - 支持多图片和视频输入"""

    # 提示词模板文件路径
    TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "提示词模板.txt")

    def __init__(self):
        self.template_content = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN_MODEL",),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "video": ("VIDEO",),
                "task_type": (["auto", "t2i", "t2v", "i2i", "i2v", "r2i", "r2v"], {
                    "default": "auto"
                }),
                "language": (["both", "chinese", "english"], {
                    "default": "both"
                }),
                "use_template": ("BOOLEAN", {
                    "default": True,
                    "label_on": "使用模板",
                    "label_off": "不使用模板"
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 128
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption_combined", "caption_chinese", "caption_english")
    FUNCTION = "generate_caption"
    CATEGORY = "QwenCLIP"

    def load_template(self):
        """加载提示词模板文件"""
        if self.template_content is None:
            try:
                if os.path.exists(self.TEMPLATE_FILE):
                    with open(self.TEMPLATE_FILE, 'r', encoding='utf-8') as f:
                        self.template_content = f.read()
                    print(f"已加载提示词模板: {self.TEMPLATE_FILE}")
                else:
                    print(f"提示词模板文件不存在: {self.TEMPLATE_FILE}")
                    self.template_content = ""
            except Exception as e:
                print(f"加载模板文件失败: {e}")
                self.template_content = ""
        return self.template_content

    def save_temp_image(self, image_tensor, index=0):
        """保存图像张量到临时文件"""
        # 转换tensor到numpy
        image_np = image_tensor[0].cpu().numpy()

        # 转换到0-255范围
        image_np = (image_np * 255).astype(np.uint8)

        # 创建PIL图像
        pil_image = Image.fromarray(image_np)

        # 保存到临时文件
        temp_dir = folder_paths.get_temp_directory()
        temp_path = os.path.join(temp_dir, f"temp_image_{index}.png")
        pil_image.save(temp_path)

        return temp_path

    def extract_frames_from_video(self, video_path, num_frames=5):
        """从视频中提取关键帧"""
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    temp_dir = folder_paths.get_temp_directory()
                    temp_path = os.path.join(temp_dir, f"temp_video_frame_{idx}.png")
                    pil_image.save(temp_path)
                    frames.append(temp_path)

            cap.release()
            return frames

        except ImportError:
            raise Exception("需要安装 opencv-python: pip install opencv-python")

    def build_prompt(self, task_type, num_images, has_video, use_template, template_content):
        """根据任务类型构建提示词"""

        # 基础提示词
        base_prompt = "请分析提供的"

        if num_images > 0:
            base_prompt += f"{num_images}张图片"
        if has_video:
            if num_images > 0:
                base_prompt += "和"
            base_prompt += "视频"

        base_prompt += "内容，生成用于AI"

        # 根据任务类型调整
        task_descriptions = {
            "t2i": "文生图",
            "t2v": "文生视频",
            "i2i": "图像编辑",
            "i2v": "图生视频",
            "r2i": "主体参考生图",
            "r2v": "主体参考生视频",
            "auto": "生成"
        }

        task_desc = task_descriptions.get(task_type, "生成")
        base_prompt += f"{task_desc}的详细提示词。"

        # 添加模板参考
        if use_template and template_content:
            # 从模板中提取相关部分
            template_sections = self.extract_template_sections(template_content, task_type)
            if template_sections:
                base_prompt += f"\n\n请参考以下模板规则：\n{template_sections}"

        # 输出格式要求
        base_prompt += """

请按照以下JSON格式返回：
{
    "中文提示词": "详细的中文描述",
    "英文提示词": "Detailed English description"
}

要求：
1. 描述要详细，包含主体、细节、风格、构图、光线等元素
2. 英文提示词要适合AI模型使用
3. 如果有多张图片，请综合分析后生成统一的提示词
4. 如果是视频内容，请描述视频中的主要场景和动作"""

        return base_prompt

    def extract_template_sections(self, template_content, task_type):
        """从模板中提取与任务类型相关的部分"""
        sections = []

        # 任务类型映射
        task_mapping = {
            "t2i": "T2I",
            "t2v": "T2V",
            "i2i": "I2I",
            "i2v": "I2V",
            "r2i": "R2I",
            "r2v": "R2V",
            "auto": None
        }

        template_task = task_mapping.get(task_type)

        # 提取通用规则
        if "电影美学设定" in template_content:
            # 提取电影美学相关规则
            start = template_content.find("电影美学设定")
            if start != -1:
                # 找到规则部分
                rule_start = template_content.find("1.", start)
                if rule_start != -1:
                    # 提取前10个规则
                    rule_end = template_content.find("生成的 prompt 示例", rule_start)
                    if rule_end == -1:
                        rule_end = min(rule_start + 2000, len(template_content))
                    rules = template_content[rule_start:rule_end].strip()
                    sections.append(f"【电影美学规则】\n{rules}")

        # 提取任务特定模板
        if template_task and template_task in template_content:
            task_start = template_content.find(template_task)
            if task_start != -1:
                # 提取该任务的描述
                task_end = template_content.find("\n\n", task_start + 100)
                if task_end == -1:
                    task_end = min(task_start + 1000, len(template_content))
                task_section = template_content[task_start:task_end].strip()
                sections.append(f"【{template_task}任务要求】\n{task_section}")

        return "\n\n".join(sections) if sections else ""

    def generate_caption(self, model, image1=None, image2=None, image3=None, image4=None,
                         image5=None, video=None, task_type="auto", language="both",
                         use_template=True, max_tokens=1024):
        """生成图像/视频描述"""

        temp_files = []  # 用于跟踪临时文件

        try:
            # 获取模型和处理器
            llm_model = model["model"]
            processor = model["processor"]

            # 收集所有输入的图片
            images = []
            for i, img in enumerate([image1, image2, image3, image4, image5], 1):
                if img is not None:
                    temp_path = self.save_temp_image(img, i)
                    images.append(temp_path)
                    temp_files.append(temp_path)

            # 处理视频输入
            video_frames = []
            if video is not None:
                # 假设video是视频文件路径
                if isinstance(video, str) and os.path.exists(video):
                    video_frames = self.extract_frames_from_video(video, num_frames=5)
                    temp_files.extend(video_frames)
                    images.extend(video_frames)

            if not images:
                raise Exception("请至少提供一张图片或视频")

            # 加载模板
            template_content = ""
            if use_template:
                template_content = self.load_template()

            # 构建提示词
            prompt = self.build_prompt(task_type, len(images) - len(video_frames),
                                       len(video_frames) > 0, use_template, template_content)

            # 构建 Qwen2.5-VL 消息格式
            content = []
            for img_path in images:
                content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            # 使用 processor 的 apply_chat_template 处理文本
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)

            # 构建输入
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            # 确定设备
            device = next(llm_model.parameters()).device
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                )

            # 裁剪掉输入部分，只保留生成的部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # 解析JSON响应
            chinese_caption = ""
            english_caption = ""

            try:
                # 尝试提取JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    json_data = json.loads(json_str)
                    chinese_caption = json_data.get('中文提示词', '')
                    english_caption = json_data.get('英文提示词', '')
                else:
                    # 如果没有JSON格式，使用原始响应
                    chinese_caption = response
                    english_caption = response

            except json.JSONDecodeError:
                print(f"警告：无法解析JSON格式的回答，使用原始文本")
                chinese_caption = response
                english_caption = response

            # 根据语言设置返回
            if language == "chinese":
                combined = chinese_caption
            elif language == "english":
                combined = english_caption
            else:
                combined = f"【中文】\n{chinese_caption}\n\n【English】\n{english_caption}"

            return (combined, chinese_caption, english_caption)

        except Exception as e:
            raise Exception(f"生成提示词失败: {str(e)}")

        finally:
            # 清理临时文件
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass


import base64
import io
import requests


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

    # 提示词模板文件路径
    TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "提示词模板.txt")

    def __init__(self):
        self.template_content = None

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
                "task_type": (["auto", "t2i", "t2v", "i2i", "i2v", "r2i", "r2v"], {
                    "default": "auto"
                }),
                "language": (["both", "chinese", "english"], {
                    "default": "both"
                }),
                "use_template": ("BOOLEAN", {
                    "default": True,
                    "label_on": "使用模板",
                    "label_off": "不使用模板"
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 128
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption_combined", "caption_chinese", "caption_english")
    FUNCTION = "generate_caption"
    CATEGORY = "QwenCLIP"

    def load_template(self):
        """加载提示词模板文件"""
        if self.template_content is None:
            try:
                if os.path.exists(self.TEMPLATE_FILE):
                    with open(self.TEMPLATE_FILE, 'r', encoding='utf-8') as f:
                        self.template_content = f.read()
                    print(f"已加载提示词模板: {self.TEMPLATE_FILE}")
                else:
                    print(f"提示词模板文件不存在: {self.TEMPLATE_FILE}")
                    self.template_content = ""
            except Exception as e:
                print(f"加载模板文件失败: {e}")
                self.template_content = ""
        return self.template_content

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

    def build_prompt(self, task_type, num_images, use_template, template_content):
        """根据任务类型构建提示词"""
        base_prompt = "请分析提供的"
        base_prompt += f"{num_images}张图片"
        base_prompt += "内容，生成用于AI"

        task_descriptions = {
            "t2i": "文生图",
            "t2v": "文生视频",
            "i2i": "图像编辑",
            "i2v": "图生视频",
            "r2i": "主体参考生图",
            "r2v": "主体参考生视频",
            "auto": "生成"
        }

        task_desc = task_descriptions.get(task_type, "生成")
        base_prompt += f"{task_desc}的详细提示词。"

        if use_template and template_content:
            template_sections = self.extract_template_sections(template_content, task_type)
            if template_sections:
                base_prompt += f"\n\n请参考以下模板规则：\n{template_sections}"

        base_prompt += """

请按照以下JSON格式返回：
{
    "中文提示词": "详细的中文描述",
    "英文提示词": "Detailed English description"
}

要求：
1. 描述要详细，包含主体、细节、风格、构图、光线等元素
2. 英文提示词要适合AI模型使用
3. 如果有多张图片，请综合分析后生成统一的提示词"""

        return base_prompt

    def extract_template_sections(self, template_content, task_type):
        """从模板中提取与任务类型相关的部分"""
        sections = []

        task_mapping = {
            "t2i": "T2I", "t2v": "T2V", "i2i": "I2I",
            "i2v": "I2V", "r2i": "R2I", "r2v": "R2V", "auto": None
        }

        template_task = task_mapping.get(task_type)

        if "电影美学设定" in template_content:
            start = template_content.find("电影美学设定")
            if start != -1:
                rule_start = template_content.find("1.", start)
                if rule_start != -1:
                    rule_end = template_content.find("生成的 prompt 示例", rule_start)
                    if rule_end == -1:
                        rule_end = min(rule_start + 2000, len(template_content))
                    rules = template_content[rule_start:rule_end].strip()
                    sections.append(f"【电影美学规则】\n{rules}")

        if template_task and template_task in template_content:
            task_start = template_content.find(template_task)
            if task_start != -1:
                task_end = template_content.find("\n\n", task_start + 100)
                if task_end == -1:
                    task_end = min(task_start + 1000, len(template_content))
                task_section = template_content[task_start:task_end].strip()
                sections.append(f"【{template_task}任务要求】\n{task_section}")

        return "\n\n".join(sections) if sections else ""

    def generate_caption(self, api_config, image1=None, image2=None, image3=None,
                         image4=None, image5=None, task_type="auto", language="both",
                         use_template=True, max_tokens=1024):
        """通过魔搭 API 生成图像描述"""
        try:
            # 收集所有输入的图片
            images = [img for img in [image1, image2, image3, image4, image5] if img is not None]

            if not images:
                raise Exception("请至少提供一张图片")

            # 加载模板
            template_content = ""
            if use_template:
                template_content = self.load_template()

            # 构建提示词
            prompt = self.build_prompt(task_type, len(images), use_template, template_content)

            # 构建消息内容
            content = []
            for img in images:
                img_url = self.image_to_base64_url(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            # 使用 requests 直接调用 API（兼容性更好）
            import requests as req

            url = f"{api_config['base_url']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_config['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": api_config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
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

            # 解析JSON响应
            chinese_caption = ""
            english_caption = ""

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
                print("警告：无法解析JSON格式的回答，使用原始文本")
                chinese_caption = result
                english_caption = result

            # 根据语言设置返回
            if language == "chinese":
                combined = chinese_caption
            elif language == "english":
                combined = english_caption
            else:
                combined = f"【中文】\n{chinese_caption}\n\n【English】\n{english_caption}"

            return (combined, chinese_caption, english_caption)

        except Exception as e:
            raise Exception(f"API 调用失败: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "QwenModelLoaderNode": QwenModelLoaderNode,
    "QwenCaptionGeneratorNode": QwenCaptionGeneratorNode,
    "ModelScopeAPILoaderNode": ModelScopeAPILoaderNode,
    "ModelScopeAPICaptionNode": ModelScopeAPICaptionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenModelLoaderNode": "加载 Qwen 模型",
    "QwenCaptionGeneratorNode": "反推提示词 (Qwen)",
    "ModelScopeAPILoaderNode": "魔搭 API 配置",
    "ModelScopeAPICaptionNode": "反推提示词 (魔搭 API)",
}
