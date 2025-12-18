from pydantic import BaseModel


class VectorStorage(BaseModel):
    name:str
    vectors:list[float]