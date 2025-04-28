import runpod
from rembg import remove, new_session
from PIL import Image
import torch
import base64
import io
import logging
import traceback
import time
import open_clip

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ 加载图片模型（从 HuggingFace Hub 加载），不能自定义
image_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:Marqo/marqo-fashionCLIP",
    precision="fp16"
)
image_model = image_model.cuda().eval()
image_processor = preprocess_val

rembg_session = new_session("u2netp")

# 打印当前GPU信息
logging.info(f"🚀 当前使用GPU: {torch.cuda.get_device_name(0)}")

# CUDA 预热 - image_model
with torch.no_grad(), torch.cuda.amp.autocast():
    dummy_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
    tensor_image = image_processor(dummy_image).unsqueeze(0).cuda()
    _ = image_model.encode_image(tensor_image, normalize=True)
logging.info("✅ 图片模型 warmup 完成")

# 核心处理函数
def handler(job):
    logging.info(f"📥 任务输入内容:\n{job}\n📄 类型: {type(job)}")
    try:
        inputs = job["input"].get("data")
        logging.info(f"📋 inputs内容是: {inputs} (类型: {type(inputs)}, 长度: {len(inputs) if inputs else 0})")
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            logging.warning("⚠️ 数据为空")
            return {
                "output": {
                    "error": "Empty input provided."
                }
            }

        start_time = time.time()

        images = []
        for i, img_str in enumerate(inputs):
            img_start_time = time.time()
            if img_str.startswith("data:image/"):
                img_str = img_str.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(img_str)))
            decode_time = time.time()
            logging.info(f"🖼️ 解码第{i}张图片耗时: {decode_time - img_start_time:.3f}s")

            image = remove(image, session=rembg_session).convert("RGB")
            rembg_time = time.time()
            logging.info(f"🧹 去背景第{i}张图片耗时: {rembg_time - decode_time:.3f}s")

            images.append(image)

        try:
            processed_images = torch.stack([image_processor(img) for img in images]).cuda()
        except Exception as e:
            logging.error(f"❌ 图片处理出错: {str(e)}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            return {
                "output": {
                    "error": f"Image processing error: {str(e)}",
                    "trace": traceback.format_exc()
                }
            }

        processor_time = time.time()
        logging.info(f"🎛️ 图片批处理耗时: {processor_time - rembg_time:.3f}s")

        with torch.no_grad(), torch.cuda.amp.autocast():
            vectors = image_model.encode_image(processed_images, normalize=True).cpu().tolist()

        inference_time = time.time()
        logging.info(f"⏱️ 图片推理耗时: {inference_time - processor_time:.3f}s")

        total_time = time.time()
        logging.info(f"✅ 总图片处理时间: {total_time - start_time:.3f}s")
        logging.info(f"✅ 推理完成，共生成 {len(vectors)} 个embedding，每个embedding维度: {len(vectors[0])}")

        return {
            "output": {
                "embeddings": vectors
            }
        }

    except Exception as e:
        logging.error(f"❌ 出现异常: {str(e)}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {
            "output": {
                "error": str(e),
                "trace": traceback.format_exc()
            }
        }

# 启动 Serverless Worker
logging.info("🟢 Image Worker 已启动，等待任务中...")
runpod.serverless.start({"handler": handler})