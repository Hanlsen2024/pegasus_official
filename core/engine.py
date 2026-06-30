"""
投票引擎 - 汇总各分析师信号，加权投票输出最终决策
"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)

# 动作常量
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


class VotingEngine:
    """加权投票引擎"""

    def __init__(self, min_consensus: float = 0.15):
        """
        Args:
            min_consensus: 最低共识分数阈值，低于此值强制 HOLD
        """
        self.min_consensus = min_consensus

    def vote(self, signals: list) -> dict:
        """
        对分析师信号进行加权投票

        Args:
            signals: [{"action": "BUY"|"SELL"|"HOLD",
                       "score": float,
                       "weight": float}, ...]

        Returns:
            {"action": str, "score": float, "details": list}
        """
        if not signals:
            logger.warning("无分析师信号，返回 HOLD")
            return {"action": HOLD, "score": 0.0, "details": []}

        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        details = []

        for sig in signals:
            w = sig.get("weight", 1.0)
            s = sig.get("score", 0.0)
            action = sig.get("action", HOLD)

            weighted = s * w
            total_weight += w

            if action == BUY:
                buy_score += weighted
            elif action == SELL:
                sell_score += weighted

            details.append({
                "action": action,
                "raw_score": round(s, 4),
                "weight": w,
                "weighted_score": round(weighted, 4),
                "reason": sig.get("reason", ""),
            })

        # 归一化
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight

        net_score = buy_score - sell_score

        # 决定最终动作
        if abs(net_score) < self.min_consensus:
            final_action = HOLD
        elif net_score > 0:
            final_action = BUY
        else:
            final_action = SELL

        logger.info(
            f"投票结果: {final_action} | "
            f"buy={buy_score:.3f} sell={sell_score:.3f} "
            f"net={net_score:.4f}"
        )

        return {
            "action": final_action,
            "score": round(net_score, 4),
            "buy_score": round(buy_score, 4),
            "sell_score": round(sell_score, 4),
            "details": details,
        }
