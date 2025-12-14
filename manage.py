from dishka import make_async_container
from core import MyProvider

async def main():
    container = make_async_container(MyProvider())
    
