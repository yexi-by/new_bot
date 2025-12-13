from pydantic import BaseModel,model_validator
from typing import Literal
from base import LLMProvider
class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"]
    text: str|None=None
    image: bytes|None=None
    
    @model_validator(mode='after')
    def check_at_least_one(self):
        if self.text is None and self.image is None:
            raise ValueError('必须提供 text 或 image')
        return self

class LLMProviderWrapper(BaseModel):
    model_name:str
    provider:LLMProvider