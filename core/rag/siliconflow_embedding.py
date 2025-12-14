import httpx

class SiliconFlowEmbedding:
    def __init__(self, apk_key: str, base_url: str, client: httpx.AsyncClient) -> None:
        self.apk_key = apk_key
        self.base_url = base_url
        self.client = client

    async def get_vector(self, model_name: str, input_text: str,**kwargs,) -> str:
        payload = {"model": model_name, "input": input_text}
        headers = {
            "Authorization": f"Bearer {self.apk_key}",
            "Content-Type": "application/json",
        }
        response = await self.client.post(self.base_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

