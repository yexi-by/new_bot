from functools import wraps
def check(func):
    @wraps(func)
    async def wrapper(*args, **kwargs)->str:
        result = await func(*args, **kwargs)
        if not result:
            raise ValueError("返回值为空")
        return result
    return wrapper

def retry_policy(retry_count:int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs)->str:
            for _ in range(retry_count):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception :
                    pass
            raise ValueError("达到最大次数")
        return wrapper
    return decorator
            