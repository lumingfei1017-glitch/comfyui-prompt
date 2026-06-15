import os
from huggingface_hub import list_repo_files

# 检查模型仓库文件结构
def check_model_files(repo_id):
    print(f"\n检查模型仓库: {repo_id}")
    try:
        # 获取仓库中的所有文件
        files = list_repo_files(repo_id)
        print(f"找到 {len(files)} 个文件")

        # 预览前20个文件
        print("\n前20个文件预览:")
        for i, file in enumerate(files[:20]):
            print(f"{i+1}. {file}")

        # 查找模型权重分块文件
        model_files = [f for f in files if (
            (f.startswith('pytorch_model-') and f.endswith('.bin')) or
            (f.startswith('model-') and f.endswith('.safetensors'))
        )]
        index_files = [f for f in files if (
            f == 'pytorch_model.bin.index.json' or
            f == 'model.safetensors.index.json'
        )]

        print(f"\n找到 {len(model_files)} 个模型权重分块文件:")
        for file in sorted(model_files):
            print(f"- {file}")

        print(f"\n找到 {len(index_files)} 个模型索引文件:")
        for file in index_files:
            print(f"- {file}")

        return model_files, index_files
    except Exception as e:
        print(f"检查模型仓库时出错: {str(e)}")
        return [], []

if __name__ == "__main__":
    # 检查Qwen2.5-VL-7B-Instruct模型
    check_model_files("Qwen/Qwen2.5-VL-7B-Instruct")