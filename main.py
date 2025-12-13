from dishka import make_async_container
from core import MyProvider
from config import Settings

container = make_async_container(MyProvider())

async def main():
    container = make_async_container(MyProvider())
    
    