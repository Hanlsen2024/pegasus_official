"""
双循环调度器 — 对标 Brale 的 Dual-Loop Architecture
- Slow Loop: K线对齐触发 → 数据+指标+AI推理
- Fast Loop: Plan Scheduler → 止盈止损监控
"""
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger("picas.scheduler")


# ---------------------------------------------------------------------------
# 交易计划
# ---------------------------------------------------------------------------

class TradePlan:
    """AI 推理生成的可执行交易计划"""

    def __init__(self, market: str, symbol: str, action: str,
                 entry_price: float, confidence: float,
                 analyst_votes: list, indicators: dict):
        self.market = market
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.confidence = confidence
        self.analyst_votes = analyst_votes
        self.indicators = indicators
        self.created_at = datetime.now()
        self.tp_levels = []   # [(价格, 平仓比例)]
        self.sl_price = None
        self.trailing_sl = None  # ATR跟踪止损价

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "action": self.action,
            "entry_price": self.entry_price,
            "confidence": self.confidence,
            "analyst_votes": self.analyst_votes,
            "created_at": self.created_at.isoformat(),
            "tp_levels": self.tp_levels,
            "sl_price": self.sl_price,
            "trailing_sl": self.trailing_sl,
        }


# ---------------------------------------------------------------------------
# 双循环调度器
# ---------------------------------------------------------------------------

class DualLoopScheduler:
    """快慢双循环调度器"""

    def __init__(self, slow_interval: int = 300, fast_interval: int = 5):
        self.slow_interval = slow_interval
        self.fast_interval = fast_interval
        self.active_plans: dict = {}       # symbol → TradePlan
        self._running = False
        self._slow_thread: Optional[threading.Thread] = None
        self._fast_thread: Optional[threading.Thread] = None
        self._slow_callback: Optional[Callable] = None
        self._on_signal: Optional[Callable] = None  # 信号回调

    def set_slow_callback(self, callback: Callable):
        """设置慢循环回调 (数据→指标→AI推理→投票)"""
        self._slow_callback = callback

    def set_signal_callback(self, callback: Callable):
        """设置信号回调 (当AI推理完成后触发)"""
        self._on_signal = callback

    def start(self):
        """启动双循环"""
        if self._running:
            return
        self._running = True

        # 慢循环线程
        self._slow_thread = threading.Thread(target=self._slow_loop, daemon=True)
        self._slow_thread.start()

        # 快循环线程
        self._fast_thread = threading.Thread(target=self._fast_loop, daemon=True)
        self._fast_thread.start()

        logger.info(f"双循环调度器已启动: slow={self.slow_interval}s fast={self.fast_interval}s")

    def stop(self):
        """停止调度器"""
        self._running = False
        logger.info("双循环调度器已停止")

    def _slow_loop(self):
        """慢循环: 数据+指标+AI推理"""
        logger.info("慢循环已启动")
        while self._running:
            try:
                if self._slow_callback:
                    result = self._slow_callback()
                    if result and self._on_signal:
                        self._on_signal(result)
            except Exception as e:
                logger.error(f"慢循环异常: {e}", exc_info=True)
            time.sleep(self.slow_interval)

    def _fast_loop(self):
        """快循环: 执行计划监控"""
        logger.info("快循环已启动")
        while self._running:
            try:
                self._check_active_plans()
            except Exception as e:
                logger.error(f"快循环异常: {e}", exc_info=True)
            time.sleep(self.fast_interval)

    def submit_plan(self, plan: TradePlan):
        """提交交易计划到快循环监控"""
        key = f"{plan.market}:{plan.symbol}"
        self.active_plans[key] = plan
        logger.info(f"计划已提交: {key} {plan.action} @{plan.entry_price}")

    def _check_active_plans(self):
        """检查所有活跃计划的止盈止损状态"""
        # 快循环检查，暂不实现实盘执行
        # 对标 Brale 的 Plan Scheduler + Freqtrade 平仓指令
        pass


# ---------------------------------------------------------------------------
# 全局调度器实例
# ---------------------------------------------------------------------------
_scheduler: Optional[DualLoopScheduler] = None


def get_scheduler() -> DualLoopScheduler:
    global _scheduler
    if _scheduler is None:
        from config.loader import load_config
        cfg = load_config().get("scheduler", {})
        _scheduler = DualLoopScheduler(
            slow_interval=cfg.get("slow_loop_interval", 300),
            fast_interval=cfg.get("fast_loop_interval", 5),
        )
    return _scheduler
