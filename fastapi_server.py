# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from contextlib import asynccontextmanager
import os
import logging
from datetime import datetime
from typing import List, Dict

# ----------------------------------------------------------------------
# Logging configuration（仅在入口配置一次，避免双打印）
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/prompt_logs.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("fastapi_server")
logger.info("HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT"))

# ----------------------------------------------------------------------
# FastAPI lifespan
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI startup: warming up model")
    pipe("你好")  # 模型预热
    yield
    logger.info("FastAPI shutdown")

app = FastAPI(lifespan=lifespan)

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=get_torch_device(),
    trust_remote_code=True,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.0,
    do_sample=False,
)

# ----------------------------------------------------------------------
# Request schema
# ----------------------------------------------------------------------
class ChatReq(BaseModel):
    model: str
    messages: List[Dict[str, str]]

# ----------------------------------------------------------------------
# Core: messages -> prompt
# ----------------------------------------------------------------------
def build_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """
    将 OpenAI-style messages 编译为单一文本 prompt
    适配 Qwen Instruct + text-generation pipeline
    """
    parts = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if not content:
            continue

        if role == "system":
            parts.append(f"系统指令：{content}")
        elif role in ("user", "human"):
            parts.append(f"用户：{content}")
        elif role in ("assistant", "ai"):
            parts.append(f"助手：{content}")
        else:
            parts.append(content)

    # 明确告诉模型：下面该你说了
    parts.append("助手：")

    return "\n".join(parts)

# ----------------------------------------------------------------------
# Chat completions API
# ----------------------------------------------------------------------
@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    logger.info("=" * 80)
    logger.info("收到请求时间: %s", datetime.now().isoformat())
    logger.info("请求模型: %s", req.model)

    logger.info("完整的 messages 结构:")
    for i, msg in enumerate(req.messages):
        logger.info("  [%d] role=%s", i, msg.get("role"))
        logger.info("      content=%s...", msg.get("content", "")[:200])

    # === 关键修复点：拼接完整 prompt ===
    prompt = build_chat_prompt(req.messages)

    logger.info("最终拼接给模型的完整 prompt:\n%s", prompt)
    logger.info("实际发送给模型的 prompt 长度: %d 字符", len(prompt))

    outputs = pipe(prompt)
    generated_text = outputs[0]["generated_text"]

    logger.info("LLM outputs 原始结构: %s", outputs)
    # 只返回 assistant 新生成的部分（可按需裁剪）
    answer = generated_text[len(prompt):].strip()
    logger.info("实际返回给客户端的 answer: \n%s", answer)

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer,
                }
            }
        ]
    }

# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
