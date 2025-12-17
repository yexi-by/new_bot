import httpx
from utils import create_retry_manager
from config import EmbeddingConfig


class SiliconFlowEmbedding:
    def __init__(
        self, embeddingconfig: EmbeddingConfig, client: httpx.AsyncClient
    ) -> None:
        self.embeddingconfig = embeddingconfig
        self.apk_key = embeddingconfig.api_key
        self.base_url = embeddingconfig.base_url
        self.client = client

    async def _get_response(
        self,
        model_name: str,
        input_text: str,
        **kwargs,
    ) -> str:
        payload = {"model": model_name, "input": input_text}
        headers = {
            "Authorization": f"Bearer {self.apk_key}",
            "Content-Type": "application/json",
        }
        response = await self.client.post(self.base_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def get_vector_representation(
        self,
        model_name: str,
        input_text: str,
        **kwargs,
    ) -> str:
        retry_count = self.embeddingconfig.retry_count
        retry_delay = self.embeddingconfig.retry_delay
        retrier = create_retry_manager(
            retry_count=retry_count,
            retry_delay=retry_delay,
            error_types=(
                httpx.HTTPStatusError,
                httpx.RequestError,
            ),
        )
        async for attempt in retrier:
            with attempt:
                response = await self._get_response(
                    model_name=model_name, input_text=input_text, **kwargs
                )
                return response
        raise RuntimeError("Retries exhausted")  # 规避下类型检查,这行是死代码
