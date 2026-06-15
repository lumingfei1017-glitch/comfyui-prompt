import os
from huggingface_hub import list_repo_files

# Qwen2.5-VL-7B-Instruct模型在Hugging Face上的仓库ID
repo_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# 列出仓库中的所有文件
try:
    files = list_repo_files(repo_id)
    print(f"成功获取 {repo_id} 仓库中的文件列表，共 {len(files)} 个文件")

    # 打印前20个文件（避免输出过多）
    print("前20个文件:")
    for i, file in enumerate(sorted(files)[:20]):
        print(f"  {i+1}. {file}")

    # 检查是否有其他类型的模型文件
    model_files = [f for f in files if any(ext in f for ext in [".safetensors", ".bin", ".pt"]) and not f.endswith(".index.json")]
    index_files = [f for f in files if ".index.json" in f]

    print(f"找到 {len(model_files)} 个模型权重文件:")
    for file in sorted(model_files):
        print(f"  - {file}")

    if index_files:
        print("找到索引文件:")
        for file in index_files:
            print(f"  - {file}")
    else:
        print("未找到索引文件")

except Exception as e:
    print(f"获取文件列表失败: {str(e)}")
    print("请确保您的网络可以访问Hugging Face，或尝试使用代理。")

    if index_files_chat:
        print("找到索引文件:")
        for file in index_files_chat:
            print(f"  - {file}")
    else:
        print("未找到索引文件")

except Exception as e:
    print(f"获取 {repo_id_chat} 文件列表失败: {str(e)}")