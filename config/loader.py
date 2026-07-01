"""
配置加载器 — 读取 config.yaml，注入环境变量
"""
import os
import yaml

_CONFIG = None
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_env(value: str) -> str:
    """解析 ${ENV_VAR} 环境变量引用"""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        return os.environ.get(var, "")
    return value


def _walk_and_resolve(data):
    """递归解析环境变量"""
    if isinstance(data, dict):
        return {k: _walk_and_resolve(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_walk_and_resolve(v) for v in data]
    else:
        return _resolve_env(data)


def load_config() -> dict:
    """加载配置 (带缓存)"""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config_path = os.path.join(_BASE_DIR, "config", "config.yaml")
    if not os.path.exists(config_path):
        # 尝试 config.example.yaml
        config_path = os.path.join(_BASE_DIR, "config", "config.example.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    _CONFIG = _walk_and_resolve(_CONFIG)
    return _CONFIG


def get_llm_config() -> dict:
    return load_config().get("llm", {})


def get_agent_config(agent_name: str) -> dict:
    return load_config().get("agents", {}).get(agent_name, {})


def get_market_config(market: str) -> dict:
    return load_config().get("markets", {}).get(market, {})


def get_risk_config(market: str = None) -> dict:
    """
    获取风控配置
    
    Args:
        market: 市场类型，crypto 时返回数字货币专属配置
    """
    cfg = load_config()
    if market == "crypto":
        crypto_cfg = cfg.get("crypto_risk", {})
        if crypto_cfg:
            return crypto_cfg
    return cfg.get("risk", {})
