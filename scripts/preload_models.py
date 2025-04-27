from huggingface_hub import snapshot_download
import os

os.environ["HF_HUB_OFFLINE"] = "0"

def download(model_id):
    print(f"📦 正在下载模型: {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=f"/runpod-volume/hub/models--{model_id.replace('/', '--')}",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ 下载完成: {model_id}")

if __name__ == "__main__":
    model = "Marqo/marqo-fashionCLIP"
    download(model)

    print("🚀 模型已完成预下载")
