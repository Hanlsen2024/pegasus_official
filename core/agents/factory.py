"""
Agent 工厂 — 统一创建和管理 Agent 实例
"""
from core.agents.indicator_agent import IndicatorAgent
from core.agents.pattern_agent import PatternAgent
from core.agents.trend_agent import TrendAgent

# Agent 注册表
AGENT_REGISTRY = {
    "indicator": IndicatorAgent,
    "pattern": PatternAgent,
    "trend": TrendAgent,
}


def create_agents(market: str = "unknown") -> list:
    """
    根据市场配置创建 AI Agent 列表

    Args:
        market: 市场类型 (gold / us_stock / a_stock)

    Returns:
        AIBaseAgent 实例列表
    """
    from config.loader import get_market_config
    market_cfg = get_market_config(market)

    active_agents = market_cfg.get("agents", ["indicator", "pattern", "trend"])
    agents = []

    for agent_name in active_agents:
        agent_cls = AGENT_REGISTRY.get(agent_name)
        if agent_cls is None:
            continue

        # 根据配置获取权重
        from config.loader import get_agent_config
        agent_cfg = get_agent_config(agent_name)
        weight = agent_cfg.get("weight", 1.0)

        agents.append(agent_cls(weight=weight))

    return agents
