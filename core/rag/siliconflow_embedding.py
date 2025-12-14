import json
import httpx
from config import Settings
class AsycnSF:
    def __init__(self,apk_key:str,base_url:str,proxy:str|None=None) -> None:
        self.apk_key=apk_key
        self.base_url=base_url
        self.proxy=proxy
        
    def get_vector_coordinates(self):
        ...
        
class SiliconFlowEmbedding:
    def __init__(self,settings:Settings) -> None:
        self.siliconflow_setting=None
        for i in settings.embedding_settings:
            if i.provider_type=="siliconflow":
                self.siliconflow_setting=i
                break
        if self.siliconflow_setting is None:
            raise ValueError("未找到硅基流动配置")
        self.api_key=self.siliconflow_setting.api_key
        self.base_url=self.siliconflow_setting.base_url
        self.model_name=self.siliconflow_setting.model_name
        
        
    async 
    
    
