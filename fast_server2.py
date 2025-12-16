from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager
import os
import logging
from datetime import datetime
from typing import List, Dict

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/prompt_logs.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("fastapi_server")
logger.info("HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT", "default"))


# ----------------------------------------------------------------------
# Model loading (极简版，避免参数冲突)
# ----------------------------------------------------------------------
def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = get_torch_device()
logger.info(f"使用设备: {DEVICE}")

# 加载tokenizer（仅保留必要配置）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
)

# 加载模型（移除所有可能冲突的参数）
logger.info(f"开始加载模型: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=DEVICE,
    trust_remote_code=True,
)
logger.info("模型加载完成")


# ----------------------------------------------------------------------
# FastAPI lifespan (移除预热，避免启动报错)
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI startup: 服务启动成功")
    yield
    logger.info("FastAPI shutdown: 服务停止")


app = FastAPI(lifespan=lifespan)


# ----------------------------------------------------------------------
# Request schema
# ----------------------------------------------------------------------
class ChatReq(BaseModel):
    model: str
    messages: List[Dict[str, str]]


# ----------------------------------------------------------------------
# Core: Prompt构建 + 模型生成（直接调用generate，绕过pipeline参数坑）
# ----------------------------------------------------------------------
def build_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """适配Qwen2.5官方格式"""
    prompt_parts = []
    system_content = ""
    user_assistant_pairs = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "system":
            system_content = """你是医学查询规范化助手，仅输出改写后的医学问题（完整单句），禁止任何额外文字，改写后立即停止。
要求：1. 仅输出改写后的问句；2. 以问号结尾（原问题有问号时）；3. 禁止任何解释、问候、追问。"""
        elif role in ("user", "human"):
            user_assistant_pairs.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role in ("assistant", "ai"):
            user_assistant_pairs.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    if system_content:
        prompt_parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")
    prompt_parts.extend(user_assistant_pairs)
    prompt_parts.append("<|im_start|>assistant\n")

    final_prompt = "".join(prompt_parts)
    logger.info(f"Prompt token数: {len(tokenizer.encode(final_prompt))}")
    return final_prompt


# ----------------------------------------------------------------------
# Chat completions API（直接调用model.generate，无参数冲突）
# ----------------------------------------------------------------------
@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    logger.info("=" * 80)
    logger.info("收到请求时间: %s", datetime.now().isoformat())
    logger.info("请求模型: %s", req.model)

    # 打印请求结构
    logger.info("完整的 messages 结构:")
    for i, msg in enumerate(req.messages):
        logger.info("  [%d] role=%s", i, msg.get("role"))
        logger.info("      content=%s...", msg.get("content", "")[:200])

    # 构建prompt
    prompt = build_chat_prompt(req.messages)
    logger.info("最终拼接给模型的完整 prompt:\n%s", prompt)

    # 模型生成（直接调用generate，绕过pipeline坑）
    try:
        # 编码prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(DEVICE)

        # 生成（仅保留核心参数，避免冲突）
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,  # 严控长度
            temperature=0.1,  # 低温度保证稳定性
            do_sample=True,  # 兼容temperature
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            repetition_penalty=1.1,  # 避免重复生成
        )

        # 解码生成结果
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        logger.info("LLM outputs 原始结构: %s", generated_text)

    except Exception as e:
        logger.error(f"模型生成失败: {str(e)}", exc_info=True)
        return {
            "choices": [{"message": {"role": "assistant", "content": ""}}]
        }

    # 结果裁剪（核心逻辑，替代stop参数）
    answer = ""
    assistant_start = "<|im_start|>assistant\n"
    last_assistant_idx = generated_text.rfind(assistant_start)

    if last_assistant_idx != -1:
        # 截取assistant生成内容
        assistant_content = generated_text[last_assistant_idx + len(assistant_start):]

        # 手动截断停止符
        stop_tokens = ["<|im_end|>", "\n<|im_start|>", "\n用户：", "\n助手："]
        for stop_token in stop_tokens:
            if stop_token in assistant_content:
                assistant_content = assistant_content.split(stop_token)[0]

        # 清洗结果
        answer = assistant_content.strip()
        answer = answer.rstrip("。，！；")

        # 补充问号（如果原问题有）
        original_user_content = ""
        for msg in reversed(req.messages):
            if msg.get("role") in ("user", "human"):
                original_user_content = msg.get("content", "").strip()
                break
        if original_user_content and (original_user_content.endswith("?") or original_user_content.endswith("？")):
            if not answer.endswith("？") and not answer.endswith("?"):
                answer += "？"

    logger.info("实际返回给客户端的 answer: \n%s", answer)

    return {
        "choices": [{"message": {"role": "assistant", "content": answer}}],
        "usage": {
            "prompt_tokens": len(inputs["input_ids"][0]),
            "completion_tokens": len(tokenizer.encode(answer)),
            "total_tokens": len(inputs["input_ids"][0]) + len(tokenizer.encode(answer))
        }
    }


# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }


# ----------------------------------------------------------------------
# 启动入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("启动FastAPI服务...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,

        reload=False,
        workers=1
    )