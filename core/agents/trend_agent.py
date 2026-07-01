"""
Trend Agent — 多周期趋势方向判断
对标 Brale: 过滤市场噪音，聚焦多时间框架结构方向
"""
from core.agents.base import AIBaseAgent


class TrendAgent(AIBaseAgent):
    """多周期趋势分析 Agent"""

    def __init__(self, weight: float = 1.0):
        super().__init__(name="📈 Trend Agent", weight=weight)

    def get_system_prompt(self, market: str) -> str:
        from config.loader import get_agent_config
        cfg = get_agent_config("trend")
        base_role = cfg.get("role", "你是一位多周期趋势分析师。")

        market_context = {
            "gold": "黄金期货通常遵循4H和日线级别的趋势。短线噪音大，需重点关注4H级别的结构方向。",
            "us_stock": "美股趋势性强，日线级别为主。关注Higher High/Higher Low结构。",
            "a_stock": "A股波动大，需关注政策驱动的结构性行情。趋势判断要注意成交量配合。",
            "crypto": "数字货币7×24小时交易，趋势延续性强但回撤剧烈。多周期共振极为重要：1H确认短期动能，4H确认中期方向，日线确认大趋势。虚假突破常见，需等待确认信号。关注合约市场资金费率作为辅助参考。",
        }.get(market, "")

        return f"""{base_role}
{market_context}
请根据多个时间框架的趋势数据，判断主趋势方向和潜在的转折信号。

关注要点：
- Higher High/Low vs Lower High/Low 结构
- 各周期的MACD方向是否一致（多周期共振）
- 均线系统的排列状态
- 关键价位附近的趋势行为"""

    def build_user_prompt(self, data_pack: dict) -> str:
        indicators = data_pack.get("indicators", {})
        multi_tf = data_pack.get("multi_timeframe", {})
        price = indicators.get("price", "N/A")
        trend = indicators.get("trend", {})

        # 多时间框架数据
        tf_lines = []
        for tf_name, tf_data in multi_tf.items():
            tf_lines.append(f"""
【{tf_name}周期】
- EMA交叉: {tf_data.get('ema_cross', 'N/A')}
- MACD方向: {tf_data.get('macd_direction', 'N/A')}
- 价格结构: {tf_data.get('structure', 'N/A')}
""")
        mtf_text = "\n".join(tf_lines) if tf_lines else "(仅单一时间框架)"

        return f"""
当前价格: {price}

【主周期趋势】
- EMA交叉: {trend.get('ema_cross')}
- 价格vs EMA50: {trend.get('price_vs_ema50')}
- MACD柱线方向: {trend.get('macd', {}).get('direction')}

【多时间框架对比】
{mtf_text}

【近期趋势结构】
- 价格位置: {indicators.get('volatility', {}).get('bollinger', {}).get('position')}
- RSI区域: {indicators.get('momentum', {}).get('rsi_zone')}

请基于多周期趋势分析，判断当前的主趋势方向和置信度。
趋势一致(多周期同向) → 高置信度
趋势分歧(不同周期方向不一) → 低置信度或观望
"""
