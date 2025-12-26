from .base import AllEvent
from pydantic import TypeAdapter, ValidationError
from typing import Any

class Event:
    def __init__(self,msg_dict:dict[str,Any]) -> None:
        self.msg_dict=msg_dict
    
    
    def get_event(self) -> AllEvent | None:
        adapter = TypeAdapter(AllEvent)
        try:
            return adapter.validate_python(self.msg_dict)
        except ValidationError:
            return None
    
