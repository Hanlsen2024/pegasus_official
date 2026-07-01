"""
Indicator Agent — 指标趋势共振分析
对标 Brale: 分析 RSI/MACD/EMA/布林带/ATR 的趋势共振，判断多周期指标一致性
"""
import json
from core.agents.base import AIBaseAgent


class IndicatorAgent(AIBaseAgent):
    """指标趋势共振分析 Agent"""

    def __init__(self, weight: float = 1.2):
        super().__init__(name="📊 Indicator Agent", weight=weight)

    def get_system_prompt(self, market: str) -> str:
        from config.loader import get_agent_config
        cfg = get_agent_config("indicator")
        base_role = cfg.get("role", "你是一位资深量化分析师。")

        market_context = {
            "gold": "当前分析对象是黄金期货(GC=F)。黄金受美元指数、实际利率、地缘政治影响，技术指标在4H周期上具有较好的参考性。",
            "us_stock": "当前分析对象是美股。美股以机构为主导，关注趋势指标和成交量确认。",
            "a_stock": "当前分析对象是A股。A股波动较大，关注超买超卖指标和资金流向。",
            "crypto": "当前分析对象是数字货币(加密货币)。7×24小时交易，波动极大，技术指标在1H/4H周期上参考性强。高波动意味着ATR值大，需放大止损容错空间。关注RSI极端值、EMA排列和多周期MACD共振。",
        }.get(market, "当前分析对象是交易标的。")

        return f"""{base_role}
{market_context}
请根据提供的多周期技术指标数据，判断是否存在「趋势共振」信号：

- EMA 排列：短中长均线是否形成多头/空头排列
- MACD：柱线方向 + 金叉/死叉状态
- RSI：是否处于超买/超卖区域
- 布林带：价格在带内的位置 + 带宽变化
- ATR：当前波动率水平

输出你的综合判断。"""

    def build_user_prompt(self, data_pack: dict) -> str:
        indicators = data_pack.get("indicators", {})
        price = indicators.get("price", "N/A")

        trend = indicators.get("trend", {})
        momentum = indicators.get("momentum", {})
        volatility = indicators.get("volatility", {})

        return f"""
当前价格: {price}

【趋势指标】
- EMA12: {trend.get('ema12')}, EMA26: {trend.get('ema26')}, EMA50: {trend.get('ema50')}
- SMA20: {trend.get('sma20')}, SMA50: {trend.get('sma50')}
- EMA交叉状态: {trend.get('ema_cross')}
- 价格vs EMA50: {trend.get('price_vs_ema50')}
- MACD柱线: {trend.get('macd', {}).get('histogram')}, 方向: {trend.get('macd', {}).get('direction')}
- MACD线: {trend.get('macd', {}).get('macd_line')}, 信号线: {trend.get('macd', {}).get('signal_line')}

【动量指标】
- RSI(14): {momentum.get('rsi14')} ({momentum.get('rsi_zone')})
- 随机指标 K: {momentum.get('stoch_k')}, D: {momentum.get('stoch_d')}
- ROC(10): {momentum.get('roc10')}%

【波动率指标】
- ATR(14): {volatility.get('atr14')} (占价格 {volatility.get('atr_pct')}%)
- 布林带上轨: {volatility.get('bollinger', {}).get('upper')}
- 布林带中轨: {volatility.get('bollinger', {}).get('mid')}
- 布林带下轨: {volatility.get('bollinger', {}).get('lower')}
- 带宽: {volatility.get('bollinger', {}).get('bandwidth_pct')}
- 价格位置: {volatility.get('bollinger', {}).get('position')}

请综合以上所有指标，判断是否存在指标共振信号，给出交易方向(BUY/SELL/HOLD)和置信度。
特别关注: EMA排列方向、MACD与RSI是否一致、布林带位置是否确认。
"""
