import asyncio
import logging
from functools import wraps

import httpx
from google.genai import errors as GErrors
from openai import (
    APIConnectionError as OAConnectionError,
)
from openai import (
    APIError as OAAPIError,
)
from openai import (
    APITimeoutError as OATimeoutError,
)
from openai import (
    OpenAIError,
)
from openai import (
    RateLimitError as OARateLimitError,
)


def check(func):
    @wraps(func)
    async def wrapper(*args, **kwargs) -> str:
        result = await func(*args, **kwargs)
        if not result:
            raise ValueError("返回值为空")
        return result

    return wrapper


def retry_policy(retry_count: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> str:
            last_exception = None
            for _ in range(retry_count):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except (
                    OARateLimitError,
                    GErrors.ClientError,
                    OAConnectionError,
                    OATimeoutError,
                    httpx.HTTPError,
                    OAAPIError,
                    GErrors.APIError,
                    OpenAIError,
                ) as e:  # 未来可能需要将这些错误区分解耦 不同错误应对不同的处理方式 待实现
                    last_exception = e
                    logging.warning(f"Retry error: {e}")
                    await asyncio.sleep(1)
            if last_exception:
                raise last_exception
            raise ValueError("达到最大次数且无具体异常信息")

        return wrapper

    return decorator


def sliding_context_window(max_context_length: int):
    if max_context_length < 2:
        raise ValueError(f"最大上下文必须大于1,当前设置: {max_context_length}")

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if len(self.messages_lst) > max_context_length:
                del self.messages_lst[1:3]
            return result

        return wrapper

    return decorator
