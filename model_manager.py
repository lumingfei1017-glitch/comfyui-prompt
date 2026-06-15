import os
import time
import requests
from pathlib import Path
import folder_paths

class ModelManager:
    def __init__(self):
        self.models_dir = os.path.join(folder_paths.models_dir, "clip")
        self.ensure_models_dir()
        
        # 模型配置
        self.model_configs = {
            "qwen2.5-vl-7b-instruct": {
                "name": "Qwen2.5-VL-7B-Instruct",
                "url": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
                "files": ["config.json", "tokenizer_config.json", "model.safetensors.index.json"] + [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
            }
        }
    
    def ensure_models_dir(self):
        """确保模型目录存在"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def get_model_path(self, model_type):
        """获取模型路径"""
        if model_type in self.model_configs:
            model_name = self.model_configs[model_type]["name"]
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_path):
                return model_path
        return None
    
    def download_model(self, model_type):
        """下载模型"""
        if model_type not in self.model_configs:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_config = self.model_configs[model_type]
        model_name = model_config["name"]
        model_path = os.path.join(self.models_dir, model_name)
        
        print(f"正在准备下载模型: {model_name}")
        
        # 创建模型目录
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # 模型下载逻辑 - 实现断点续传
        print(f"请稍候，模型 {model_name} 正在从Hugging Face下载...")
        
        # 下载主要模型文件
        try:
            # 下载配置文件
            self._download_file_with_resume(
                f"https://huggingface.co/qwen/{model_name}/raw/main/config.json",
                os.path.join(model_path, "config.json")
            )
            print(f"配置文件已下载")
            
            # 下载tokenizer配置
            self._download_file_with_resume(
                f"https://huggingface.co/qwen/{model_name}/raw/main/tokenizer_config.json",
                os.path.join(model_path, "tokenizer_config.json")
            )
            print(f"tokenizer配置已下载")
            
            # 下载模型索引文件
            index_file = model_config['files'][2]  # 索引文件是files列表中的第三个元素
            self._download_file_with_resume(
                f"https://huggingface.co/Qwen/{model_name}/resolve/main/{index_file}",
                os.path.join(model_path, index_file)
            )
            print(f"模型索引文件已下载")
            
            # 下载模型权重文件（分块）
            # 权重文件是files列表中从第四个元素开始的所有元素
            weight_files = model_config['files'][3:]
            for i, file_name in enumerate(weight_files, 1):
                self._download_file_with_resume(
                    f"https://huggingface.co/Qwen/{model_name}/resolve/main/{file_name}",
                    os.path.join(model_path, file_name)
                )
                print(f"模型权重文件{i}已下载")
            
        except Exception as e:
            # 尝试使用国内镜像
            try:
                print(f"尝试使用国内镜像下载...")
                # 下载配置文件
                self._download_file_with_resume(
                    f"https://hf-mirror.com/Qwen/{model_name}/raw/main/config.json",
                    os.path.join(model_path, "config.json")
                )
                print(f"配置文件已下载")
                
                # 下载tokenizer配置
                self._download_file_with_resume(
                    f"https://hf-mirror.com/Qwen/{model_name}/raw/main/tokenizer_config.json",
                    os.path.join(model_path, "tokenizer_config.json")
                )
                print(f"tokenizer配置已下载")
                
                # 下载模型索引文件
                index_file = model_config['files'][2]  # 索引文件是files列表中的第三个元素
                self._download_file_with_resume(
                    f"https://hf-mirror.com/Qwen/{model_name}/resolve/main/{index_file}",
                    os.path.join(model_path, index_file)
                )
                print(f"模型索引文件已下载")
                
                # 下载模型权重文件（分块）
                # 权重文件是files列表中从第四个元素开始的所有元素
                weight_files = model_config['files'][3:]
                for i, file_name in enumerate(weight_files, 1):
                    self._download_file_with_resume(
                        f"https://hf-mirror.com/Qwen/{model_name}/resolve/main/{file_name}",
                        os.path.join(model_path, file_name)
                    )
                    print(f"模型权重文件{i}已下载")
            except Exception as e2:
                error_msg = f"模型下载失败: {str(e2)}"
                error_msg += "\n请确保您的网络连接正常，并且可以访问Hugging Face网站或国内镜像。"
                error_msg += "\n如果下载持续失败，您可以尝试手动下载模型文件并放置到 {model_path} 目录下。"
                error_msg += f"\n模型下载地址: https://huggingface.co/Qwen/{model_name}"
                raise Exception(error_msg)

        return model_path
    
    def _download_file_with_resume(self, url, file_path, chunk_size=1024*10, max_retries=3):
        """带断点续传的文件下载方法"""
        file_size = 0
        retries = 0
        
        # 检查文件是否已部分下载
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"发现已下载部分文件，尝试续传: {file_path} ({file_size} bytes)")
        
        while retries <= max_retries:
            try:
                headers = {'Range': f'bytes={file_size}-'}
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                
                # 检查响应状态
                if response.status_code == 200 or response.status_code == 206:
                    total_size = int(response.headers.get('content-length', 0)) + file_size
                    
                    with open(file_path, 'ab') as f:
                        downloaded = file_size
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # 显示下载进度
                                if total_size > 0:
                                    progress = downloaded / total_size * 100
                                    print(f"下载进度: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
                    
                    print()  # 新行
                    return
                else:
                    raise Exception(f"下载失败，状态码: {response.status_code}")
            except Exception as e:
                retries += 1
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"下载失败 (尝试 {retries}/{max_retries}): {str(e)}")
                if retries > max_retries:
                    raise e
                print(f"5秒后重试...")
                time.sleep(5)

    def is_model_downloaded(self, model_type):
        """检查模型是否已下载"""
        model_path = self.get_model_path(model_type)
        return model_path is not None and os.path.exists(model_path)
