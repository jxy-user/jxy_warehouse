import os
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, Set

from fastapi import HTTPException, Request, status


_RATE_LIMITER_LOCK = Lock()
_RATE_LIMITER_BUCKETS: Dict[str, Deque[float]] = defaultdict(deque)


def load_api_keys() -> Set[str]:
    """
    从环境变量读取可用 API Key，格式:
    MEDFUSE_API_KEYS=dev-key,demo-key
    """
    raw = os.getenv("MEDFUSE_API_KEYS", "")
    keys = {x.strip() for x in raw.split(",") if x.strip()}
    return keys


def _extract_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def verify_api_key(request: Request, enabled: bool = True) -> str:
    if not enabled:
        return "security-disabled"

    keys = load_api_keys()
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务端未配置API Key，请联系管理员设置 MEDFUSE_API_KEYS",
        )

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少X-API-Key请求头",
        )
    if api_key not in keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key无效",
        )
    return api_key


def check_rate_limit(
    request: Request,
    api_key: str,
    rate_limit_per_minute: int = 60,
    enabled: bool = True,
) -> None:
    if not enabled:
        return

    now = time.time()
    window_seconds = 60.0
    client_ip = _extract_client_ip(request)
    bucket_key = f"{api_key}|{client_ip}"

    with _RATE_LIMITER_LOCK:
        bucket = _RATE_LIMITER_BUCKETS[bucket_key]
        while bucket and (now - bucket[0]) > window_seconds:
            bucket.popleft()
        if len(bucket) >= rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，限制为每分钟 {rate_limit_per_minute} 次",
            )
        bucket.append(now)


def require_auth_and_rate_limit(
    request: Request,
    enabled: bool = True,
    rate_limit_per_minute: int = 60,
) -> str:
    api_key = verify_api_key(request=request, enabled=enabled)
    check_rate_limit(
        request=request,
        api_key=api_key,
        rate_limit_per_minute=rate_limit_per_minute,
        enabled=enabled,
    )
    return api_key
