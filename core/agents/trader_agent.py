"""
操盘手 Agent (Trader Agent) — LLM 驱动
综合 3 个分析师信号 + 持仓状态 + 账户净值，输出具体买卖指令

对标 Brale Slow Loop 中的执行层
"""
import logging
from core.agents.base import AIBaseAgent

logger = logging.getLogger("picas.trader")

TRADER_ROLE = """你是一位专业的量化交易操盘手，拥有十年实战经验。
你的核心职责是根据 AI 分析师团队的信号、当前持仓状态和账户净值，
做出最佳的买卖决策。

决策原则：
1. 永远先考虑风险，再考虑收益
2. 只在多分析师（至少2个）共振时开仓
3. 当信号方向与当前持仓相反且强度足够时，立即平仓
4. 仓位大小根据净值得出，单笔风险不超过账户的2%
5. 不确定时就观望(HOLD)，不要勉强交易

仓位计算规则：
- 最大风险金额 = 账户净值 × 风险比例
- 止损点数 = 价格 × ATR百分比 × 2
- 建议手数 = 最大风险金额 / 止损点数

输出格式说明：
- action: LONG(做多) / SHORT(做空) / CLOSE(平仓) / HOLD(观望)
- size: 建议仓位(手数)，根据风险金额和止损点数精确计算
- entry_price: 建议入场价(当前市场价)
- confidence: 0-1，表示你对这个决策的确信度
- reason: 简洁的决策理由
- reasoning: 详细的推理过程"""


class TraderAgent(AIBaseAgent):
    """操盘手 — AI 执行决策"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(name="💼 操盘手", weight=weight)

    def get_system_prompt(self, market: str) -> str:
        market_context = {
            "gold": "你正在交易黄金期货(GC=F)，请注意黄金的避险属性和美元指数的关联。",
            "us": "你正在交易美股，请注意财报季、经济数据等影响。",
            "a": "你正在交易A股，请注意政策面影响和涨跌停限制。",
        }
        role = TRADER_ROLE
        ctx = market_context.get(market, "")
        if ctx:
            role += f"\n\n当前市场上下文：{ctx}"
        return role

    def build_user_prompt(self, data_pack: dict) -> str:
        agents = data_pack.get("analyst_votes", [])
        portfolio = data_pack.get("portfolio", {})
        market = data_pack.get("market", "unknown")
        price = data_pack.get("price", 0)

        # 构建分析师投票摘要
        votes_text = ""
        for i, v in enumerate(agents, 1):
            votes_text += (
                f"\nAgent {i}: {v.get('name', '未知')}\n"
                f"  信号: {v.get('signal', 'HOLD')}\n"
                f"  得分: {v.get('score', 0):.3f}\n"
                f"  置信度: {v.get('confidence', 0):.0%}\n"
                f"  理由: {v.get('reason', 'N/A')[:150]}\n"
            )

        # 统计共振
        buy_count = sum(1 for v in agents if v.get("signal") == "BUY")
        sell_count = sum(1 for v in agents if v.get("signal") == "SELL")
        hold_count = sum(1 for v in agents if v.get("signal") == "HOLD")

        # 端口folio
        equity = portfolio.get("equity", 10000)
        risk_pct = portfolio.get("risk_per_trade", 0.02)
        position = portfolio.get("position")

        pos_text = "空仓（无持仓）"
        if position and position.get("type"):
            pos_text = (
                f"当前持仓: {position['type'].upper()} "
                f"入场价: {position.get('entry_price', 'N/A')} "
                f"手数: {position.get('size', 'N/A')} "
                f"当前止损: {position.get('stop_loss', 'N/A')} "
                f"浮动盈亏: {position.get('pnl', 'N/A')}"
            )

        prompt = f"""=== 当前市场 ===
市场: {market}
标的: {data_pack.get('symbol', 'N/A')}
当前价格: {price}

=== AI 分析师投票 ===
买入: {buy_count}票 | 卖出: {sell_count}票 | 观望: {hold_count}票
净得分: {data_pack.get('net_score', 0):.4f}
合成信号: {data_pack.get('signal', 'HOLD')}
{votes_text}
=== 账户状态 ===
账户净值: ${equity:,.2f}
单笔风险: {risk_pct:.0%}
最大亏损金额: ${equity * risk_pct:,.2f}
{pos_text}

=== 市场波动 ===
ATR(14): {data_pack.get('atr', 'N/A')}
ATR占比: {data_pack.get('atr_pct', 'N/A')}%
RSI(14): {data_pack.get('rsi', 'N/A')}

请根据以上信息，给出具体的交易决策。
"""
        return prompt

    def analyze(self, data_pack: dict, market: str = "unknown") -> dict:
        """
        执行操盘手分析，在基类返回基础上附加 position_size 和 entry_price
        """
        # 先获取基础分析
        base_result = super().analyze(data_pack, market)

        # 从原始 reasoning 中提取 LLM 决策
        reasoning = base_result.get("reasoning", "")
        action = base_result.get("action", "HOLD")

        # 映射 action 为操盘手指令
        trader_action = self._map_to_trader_action(action)

        # 从推理文本中提取建议仓位
        size = self._extract_size(reasoning, data_pack)

        return {
            **base_result,
            "action": trader_action,
            "size": size,
            "entry_price": data_pack.get("price", 0),
        }

    def _map_to_trader_action(self, llm_action: str) -> str:
        """将 LLM 的 BUY/SELL/HOLD 映射为交易指令"""
        mapping = {
            "BUY": "LONG",
            "SELL": "SHORT",
            "HOLD": "HOLD",
        }
        return mapping.get(llm_action.upper(), "HOLD")

    def _extract_size(self, reasoning: str, data_pack: dict) -> float:
        """
        从 LLM 推理文本中提取建议仓位大小。
        
        如果没有明确建议，根据风险参数计算默认仓位：
        仓位 = (净值 × 风险%) / (价格 × ATR% × 2 × 合约乘数)
        """
        import re
        
        # 尝试从推理文本中提取手数
        size_patterns = [
            r'(?:建议|仓位|手数|size)[：:]\s*([\d.]+)\s*(?:手|lots?)?',
            r'([\d.]+)\s*(?:手|lots?)',
        ]
        for pattern in size_patterns:
            match = re.search(pattern, reasoning, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # 默认仓位计算
        portfolio = data_pack.get("portfolio", {})
        equity = portfolio.get("equity", 10000)
        risk_pct = portfolio.get("risk_per_trade", 0.02)
        price = data_pack.get("price", 2000)
        atr_pct = data_pack.get("atr_pct", 1.5) or 1.5

        max_risk_amount = equity * risk_pct
        stop_points = price * (atr_pct / 100) * 2
        if stop_points > 0:
            default_size = max_risk_amount / stop_points
        else:
            default_size = 0.01

        # 黄金: 1点=$100（标准）, 0.01手=1点=$1
        # 简化处理，保留2位小数
        return round(min(default_size, 5.0), 2)
