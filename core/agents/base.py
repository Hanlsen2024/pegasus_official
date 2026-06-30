"""
AI Agent 基类 — 对标 Brale 的 Agent 抽象
每个 Agent 绑定一个 LLM 实例，独立推理
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("picas.agent")


class AIBaseAgent(ABC):
    """AI 驱动的分析Agent基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self._llm = None

    @property
    def llm(self):
        """懒加载 LLM 客户端"""
        if self._llm is None:
            from core.agents.llm_client import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    @abstractmethod
    def get_system_prompt(self, market: str) -> str:
        """获取系统提示词 (Agent角色定义)"""
        ...

    @abstractmethod
    def build_user_prompt(self, data_pack: dict) -> str:
        """构建用户提示词 (注入指标/数据)"""
        ...

    def analyze(self, data_pack: dict, market: str = "unknown") -> dict:
        """
        执行 AI 分析

        Args:
            data_pack: 包含 indicators / sr_levels / price 等
            market: 市场类型

        Returns:
            {"action": "BUY"|"SELL"|"HOLD", "score": float, "reason": str,
             "reasoning": str, "signals": list, "confidence": float}
        """
        try:
            system_prompt = self.get_system_prompt(market)
            user_prompt = self.build_user_prompt(data_pack)

            logger.info(f"[{self.name}] 发起 LLM 推理...")
            result = self.llm.chat_structured(system_prompt, user_prompt)

            action = result.get("action", "HOLD")
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")
            signals = result.get("signals", [])

            # 根据 action 和 confidence 计算得分
            if action == "BUY":
                score = confidence * 0.8
            elif action == "SELL":
                score = -confidence * 0.8
            else:
                score = 0.0

            logger.info(f"[{self.name}] → {action} conf={confidence:.2f} score={score:.3f}")

            return {
                "action": action,
                "score": round(score, 4),
                "confidence": confidence,
                "reason": reasoning[:120],
                "reasoning": reasoning,
                "signals": signals,
                "key_level": result.get("key_level", ""),
            }

        except Exception as e:
            logger.error(f"[{self.name}] 分析异常: {e}", exc_info=True)
            return {
                "action": "HOLD", "score": 0.0, "confidence": 0.0,
                "reason": f"Agent异常: {e}",
                "reasoning": "", "signals": [], "key_level": "",
            }
