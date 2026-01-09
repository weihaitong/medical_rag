import os
import torch
import logging
import time
import requests
import inspect
from typing import Optional, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Phi3ForCausalLM,
    pipeline
)
# 假设 ollama.py 在同一级目录
from ollama import OllamaClient

logger = logging.getLogger(__name__)


def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_dtype():
    if torch.cuda.is_available():
        return torch.float16
    elif torch.backends.mps.is_available():
        return torch.float16
    else:
        return torch.float32


# === Rewrite LLM 辅助类 (原逻辑复原) ===
class FastAPIRewriteLLM(Runnable):
    def __init__(self, endpoint: str, model: str, timeout: float = 2.0):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def invoke(self, input, **kwargs) -> str:
        if isinstance(input, ChatPromptValue):
            messages = [{"role": m.type, "content": m.content} for m in input.to_messages()]
        else:
            messages = [{"role": "user", "content": str(input)}]

        payload = {"model": self.model, "messages": messages}
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"FastAPI call failed: {e}")


class RewriteWithFallback(Runnable):
    def __init__(self, remote_llm, local_llm):
        self.remote_llm = remote_llm
        self.local_llm = local_llm

    def invoke(self, input, config=None, **kwargs) -> str:
        # 1. 优先调用远端
        try:
            start = time.time()
            content = self.remote_llm.invoke(input)
            latency = time.time() - start
            logger.info(f"Rewrite remote success, latency={latency:.2f}s")
            if isinstance(content, BaseMessage):
                return content.content.strip()
            return str(content).strip()
        except Exception as e:
            # 2. 降级本地
            if isinstance(input, ChatPromptValue):
                local_prompt = input.to_string()
            else:
                local_prompt = str(input)
            logger.warning(f"Rewrite fallback to local, reason={e}, query={local_prompt}")
            return self.local_llm.invoke(local_prompt).strip()


class ModelFactory:
    """模型工厂：负责所有模型的加载与持有"""

    def __init__(self):
        self.device = get_torch_device()
        self.dtype = get_dtype()

        # 模型实例
        self.embedding_model = None
        self.reranker = None
        self.llm = None
        self.rewrite_llm = None
        self.ollama = None

        # 初始化流程
        self._init_ollama()
        self._init_models()
        self._init_rewrite_llm()

    def _init_ollama(self):
        logger.info("初始化 OllamaClient...")
        self.ollama = OllamaClient(
            model="qwen2:7b-instruct-q5_K_M",
            base_url="http://localhost:11434",
            timeout=120,
        )

    def _init_models(self):
        # 1. Embedding
        logger.info("加载嵌入模型 (BAAI/bge-m3)...")
        try:
            model_path = "models/bge-m3"
            target_model = model_path if os.path.exists(model_path) else "BAAI/bge-m3"
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=target_model,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}", exc_info=True)
            logger.warning("→ 使用轻量级备用模型")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": self.device},
            )

        # 2. Reranker
        logger.info("加载重排器 (BAAI/bge-reranker-v2-m3)...")
        try:
            reranker_path = "models/bge-reranker-v2-m3"
            target_model = reranker_path if os.path.exists(reranker_path) else "BAAI/bge-reranker-v2-m3"
            self.reranker_args = {
                "model_name": target_model,
                "device": "cpu",  # 强制使用cpu解决mac在这个模型的bug
                "model_kwargs": {
                    "trust_remote_code": True,
                    # --- 修复开始 ---
                    # 必须通过 automodel_args 传递底层模型参数
                    "automodel_args": {
                        "torch_dtype": torch.float32
                    }
                }
            }
            self.reranker = HuggingFaceCrossEncoder(
                model_name=self.reranker_args["model_name"],
                model_kwargs = self.reranker_args["model_kwargs"]
            )
        except Exception as e:
            logger.error(f"重排器加载失败: {e}", exc_info=True)
            logger.warning("→ 使用备用重排器")
            self.reranker = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

        # 3. Main LLM
        logger.info("加载本地 LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
        try:
            llm_path = "models/qwen2.5-0.5b"
            if os.path.exists(llm_path):
                logger.info(f"→ 使用本地LLM: {llm_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    llm_path, torch_dtype=self.dtype, device_map="auto", trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
            else:
                model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                logger.info(f"→ 下载 LLM: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=self.dtype, device_map="auto", trust_remote_code=True
                )

            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=256, temperature=0.0, do_sample=False, repetition_penalty=1.2
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.error(f"LLM加载失败: {e}", exc_info=True)
            # 备用 Phi-3 逻辑 (简化保留关键逻辑)
            logger.warning("→ 尝试使用 Phi-3...")
            try:
                model = Phi3ForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct", device_map="auto", torch_dtype="auto", trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256,
                                temperature=0.3)
                self.llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as e2:
                logger.critical(f"备用LLM加载失败: {e2}")
                raise

    def _init_rewrite_llm(self):
        """Query Rewrite LLM: FastAPI优先 + 本地Fallback"""
        # 本地 fallback (复用主 LLM)
        local_llm = self.llm

        try:
            remote_llm = FastAPIRewriteLLM(
                endpoint="http://127.0.0.1:8000/v1/chat/completions",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                timeout=2.0
            )
            self.rewrite_llm = RewriteWithFallback(remote_llm, local_llm)
            logger.info("Rewrite LLM 使用策略：FastAPI 远程优先 + 本地 fallback")
        except Exception as e:
            logger.warning(f"FastAPI Rewrite LLM 初始化失败: {e}")
            self.rewrite_llm = local_llm