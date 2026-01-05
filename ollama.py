# ollama.py
import logging
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    负责与本地 Ollama 服务通信的客户端（基于 /api/chat 协议）
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        default_options: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_options = default_options or {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 1024,
        }

        logger.info(
            "OllamaClient initialized: model=%s, base_url=%s",
            self.model,
            self.base_url,
        )

    # 【修改点 1】增加了 model 参数，默认为 None
    def generate(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
        """
        同步调用 Ollama /api/chat
        messages: [{"role": "system|user|assistant", "content": "文本内容"}]
        model: (可选) 动态覆盖默认模型
        """
        url = f"{self.base_url}/api/chat"

        # 【修改点 2】优先使用传入的 model，否则用 self.model
        target_model = model if model else self.model

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": self._merge_options(options),
        }

        logger.debug("Ollama request payload=%s", payload)

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Ollama HTTP 调用失败", exc_info=True)
            raise RuntimeError("Ollama 服务不可用") from e

        data = response.json()

        # Ollama /api/chat 返回结构通常是 { "model": ..., "message": { "role": ..., "content": ... }, ... }
        if "message" not in data or "content" not in data["message"]:
            logger.error("Ollama 返回异常格式: %s", data)
            raise ValueError("Ollama 返回数据中缺少 message.content 字段")

        return data["message"]["content"].strip()

    def _merge_options(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self.default_options)
        if options:
            merged.update(options)
        return merged