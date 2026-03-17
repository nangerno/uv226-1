import json
import os
import redis

STATE_KEY = "state"

# In-memory fallback when Redis is unavailable (e.g. local dev)
_redis_unavailable: bool = False
_local_state: dict = {}


def _get_redis_client() -> redis.Redis | None:
    """Get a Redis client, or None if Redis is disabled/unavailable."""
    if _redis_unavailable:
        return None
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD", None)
    db = int(os.getenv("REDIS_DB", 0))
    return redis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=True,
    )


def get_state() -> dict:
    """Return the state from Redis, or in-memory fallback if Redis is unavailable."""
    global _redis_unavailable, _local_state
    if _redis_unavailable:
        return dict(_local_state)
    try:
        client = _get_redis_client()
        if client is None:
            return dict(_local_state)
        value = client.get(STATE_KEY)
        if value is None:
            return {}
        return json.loads(value)
    except (redis.exceptions.ConnectionError, OSError):
        _redis_unavailable = True
        return dict(_local_state)
    except json.JSONDecodeError:
        return {}


def set_state(state: dict) -> None:
    """Set the state in Redis, or in-memory fallback if Redis is unavailable."""
    global _redis_unavailable, _local_state
    if _redis_unavailable:
        _local_state.clear()
        _local_state.update(state)
        return
    try:
        client = _get_redis_client()
        if client is None:
            _local_state.clear()
            _local_state.update(state)
            return
        client.set(STATE_KEY, json.dumps(state))
    except (redis.exceptions.ConnectionError, OSError):
        _redis_unavailable = True
        _local_state.clear()
        _local_state.update(state)


def test():
    state = get_state()
    print(json.dumps(state, indent=4, ensure_ascii=False))
    
if __name__ == "__main__":
    test()