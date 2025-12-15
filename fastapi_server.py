# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from contextlib import asynccontextmanager

# app = FastAPI()
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    pipe("你好")  # 模型预热
    yield
    # shutdown（可选）

app = FastAPI(lifespan=lifespan)

def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    else:
        return "cpu"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=get_torch_device(),
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.0,
    do_sample=False
)

class ChatReq(BaseModel):
    model: str
    messages: list

@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    prompt = req.messages[-1]["content"]
    out = pipe(prompt)[0]["generated_text"]
    return {
        "choices": [
            {"message": {"role": "assistant", "content": out}}
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

