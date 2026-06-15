import os
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO

class ImageCaptionGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_type = None
        self.device = torch.device("cpu")

    def load_model(self, model_path, model_type):
        """加载 Qwen2.5-VL 视觉语言模型"""
        try:
            if self.model is not None and self.model_type == model_type:
                return

            # 卸载现有模型
            self.unload_model()

            print(f"正在加载模型: {model_path}")

            # 加载处理器（Qwen2.5-VL 使用 Processor 而非 Tokenizer）
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # 创建临时卸载目录
            offload_folder = os.path.join(os.path.dirname(model_path), "offload")
            os.makedirs(offload_folder, exist_ok=True)

            # 尝试从本地路径加载
            try:
                print(f"尝试以 Qwen2_5_VLForConditionalGeneration 加载 {model_path} ...")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    device_map="auto",
                    offload_folder=offload_folder,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
                print(f"模型已加载，部分权重卸载到: {offload_folder}")
            except Exception as e:
                # 如果本地失败，尝试从 Hugging Face Hub 下载
                print(f"本地模型加载失败: {str(e)}")
                print(f"尝试从 Hugging Face Hub 下载: {model_path}")
                model_name = os.path.basename(model_path)
                hub_model_id = f"Qwen/{model_name}" if not model_path.startswith("Qwen/") else model_path

                offload_folder = os.path.join(os.path.dirname(model_path), "offload")
                os.makedirs(offload_folder, exist_ok=True)

                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    hub_model_id,
                    device_map="auto",
                    offload_folder=offload_folder,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
                print(f"模型已从 Hugging Face Hub 加载，部分权重卸载到: {offload_folder}")

            self.model_type = model_type
            print("模型加载完成")

        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")

    def unload_model(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_type = None
        print("模型已卸载")
    
    def generate_caption(self, image_path, detail_level):
        """生成图像描述（中英文）"""
        if self.model is None or self.processor is None:
            raise Exception("模型未加载")

        try:
            # 构建中文提示词，明确说明用于AI文生图
            prompt = "请给我一段提示词，可以准确向其他文生图大模型描述这张图片，以生成相似的图片，返回文本需要包含中英文，给出json格式回答，具体内容是{\"中文提示词\":\"\",\"英文提示词\":\"\"}，描述内容尽可能详细，可能包括但不限于主体（含权重）、位置关系、细节、风格等"

            # 构建 Qwen2.5-VL 消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # 使用 processor 的 apply_chat_template 处理文本
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)

            # 构建输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                )

            # 裁剪掉输入部分，只保留生成的部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # 解析 JSON 响应
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    json_data = json.loads(json_str)
                    chinese_caption = json_data.get('中文提示词', '')
                    english_caption = json_data.get('英文提示词', '')
                else:
                    chinese_caption = response
                    english_caption = response
            except json.JSONDecodeError:
                chinese_caption = response
                english_caption = response
                print(f"警告：无法解析JSON格式的回答，使用原始文本。原始回答: {response}")

            # 返回中英文版本
            return chinese_caption, english_caption

        except Exception as e:
            raise Exception(f"生成描述失败: {str(e)}")
    
    def __del__(self):
        """析构函数，确保模型被卸载"""
        self.unload_model()
