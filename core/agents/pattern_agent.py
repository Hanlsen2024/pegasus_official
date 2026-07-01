"""
Pattern Agent — 形态识别与价格行为分析
对标 Brale: 识别蜡烛图形态、支撑阻力位、SMC流动性区域
"""
from core.agents.base import AIBaseAgent


class PatternAgent(AIBaseAgent):
    """形态与价格行为分析 Agent"""

    def __init__(self, weight: float = 1.0):
        super().__init__(name="🕯️ Pattern Agent", weight=weight)

    def get_system_prompt(self, market: str) -> str:
        from config.loader import get_agent_config
        cfg = get_agent_config("pattern")
        base_role = cfg.get("role", "你是一位价格行为专家。")

        market_context = {
            "crypto": "数字货币市场7×24小时交易，K线形态和SMC流动性区域在1H/4H周期极为有效。特别注意：假突破频繁，需等待收盘确认；流动性猎杀(插针)常见于关键价位；整数关口和心理价位(如BTC 10万)有强磁吸效应。",
        }.get(market, "")

        return f"""{base_role}
{market_context}
请根据提供的近期价格走势数据、支撑阻力位和成交量分布，识别以下信号：

- 蜡烛图形态：锤子线/倒锤子、吞没、晨星/夜星、十字星
- 支撑阻力：价格是否在关键S/R位附近
- 成交量确认：突破是否带量、回调是否缩量
- 价格行为：Higher High/Low 还是 Lower High/Low 结构

重点关注价格在关键位置的行为，而非单纯的指标数值。"""

    def build_user_prompt(self, data_pack: dict) -> str:
        indicators = data_pack.get("indicators", {})
        sr = indicators.get("sr_levels", {})
        price = indicators.get("price", "N/A")
        volume = indicators.get("volume", {})

        # 近期价格走势摘要
        price_summary = data_pack.get("price_summary", "")

        return f"""
当前价格: {price}

【支撑阻力位】
- 阻力位: {sr.get('resistance')} (近期高点)
- 支撑位: {sr.get('support')} (近期低点)
- 枢轴点: {sr.get('pivot')}
- 摆动高点: {sr.get('swing_highs', [])}
- 摆动低点: {sr.get('swing_lows', [])}

【成交量分布】
- OBV趋势: {volume.get('obv_trend', 'N/A')}
- 成交量最大价位(POC): {volume.get('poc', 'N/A')}
- 价值区域: {volume.get('val', 'N/A')} ~ {volume.get('vah', 'N/A')}

【近期价格走势】
{price_summary}

请基于以上数据识别：
1. 当前价格在支撑阻力体系中的位置
2. 是否有可识别的蜡烛图形态信号
3. 成交量是否验证价格行为
4. 综合给出交易方向(BUY/SELL/HOLD)和置信度
"""
