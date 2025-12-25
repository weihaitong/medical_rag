import os
import torch
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ollama import OllamaClient
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Phi3ForCausalLM,
    pipeline
)

logger = logging.getLogger(__name__)

class MedicalModelFactory:
    """
    负责加载和管理所有 AI 模型实例 (单例模式推荐)
    """
    def __init__(self):
        self.device = self._get_torch_device()
        self.dtype = self._get_dtype()

    def _get_torch_device(self):
        if torch.cuda.is_available(): return "cuda"
        elif torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def _get_dtype(self):
        if self.device in ["cuda", "mps"]: return torch.float16
        return torch.float32

    def load_embedding_model(self):
        logger.info("正在加载 Embedding 模型...")
        # 这里保留你原本的加载逻辑
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3", # 或本地路径
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

    def load_reranker(self):
        logger.info("正在加载 Rerank 模型...")
        return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

    def load_ollama_client(self):
        logger.info("初始化 Ollama 客户端...")
        return OllamaClient(
            model="qwen2:7b-instruct-q5_K_M",
            base_url="http://localhost:11434",
            timeout=120
        )

    def load_local_llm(self):
        """加载本地兜底 LLM (如 0.5B 模型)"""
        logger.error(f"LLM加载失败: {e}", exc_info=True)
        logger.warning("→ 使用超轻量级模型: microsoft/Phi-3-mini-4k-instruct (需要在线下载)")
        try:
            model = Phi3ForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.3
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e2:
            logger.error(f"备用LLM加载失败: {e2}", exc_info=True)
            logger.critical("无法加载任何语言模型。请检查网络连接或创建 models/ 目录并放置本地模型。")
            raise
        pass