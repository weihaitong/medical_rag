import requests
from langchain_core.runnables import Runnable
from langchain_core.prompt_values import ChatPromptValue

class FastAPIRewriteLLM(Runnable):
    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float = 2.0,
    ):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def invoke(self, input: str, **kwargs) -> str:
        if isinstance(input, ChatPromptValue):
            messages = input.to_messages()
            input = messages[-1].content

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": input}
            ]
        }

        resp = requests.post(
            self.endpoint,
            json=payload,
            timeout=(1.0, 10.0)  # (connect_timeout, read_timeout)
        )
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]

# curl -X POST "http://localhost:8000/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "your-model-name",
#     "messages": [
#       {"role": "user", "content": "你好，请介绍一下你自己。"}
#     ]
#   }'