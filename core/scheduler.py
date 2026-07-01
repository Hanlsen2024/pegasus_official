"""
双循环调度器 — 对标 Brale 的 Dual-Loop Architecture
- Slow Loop: K线对齐触发 → 数据+指标+AI推理
- Fast Loop: Plan Scheduler → 止盈止损实时监控
"""
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Callable, Dict, List

logger = logging.getLogger("picas.scheduler")


# ---------------------------------------------------------------------------
# 交易计划 (增强版 — 支持完整风控数据)
# ---------------------------------------------------------------------------

class TradePlan:
    """AI 推理生成的可执行交易计划"""

    def __init__(self, market: str, symbol: str, action: str,
                 entry_price: float, confidence: float,
                 analyst_votes: list, indicators: dict):
        self.market = market
        self.symbol = symbol
        self.action = action          # LONG / SHORT
        self.entry_price = entry_price
        self.confidence = confidence
        self.analyst_votes = analyst_votes
        self.indicators = indicators
        self.created_at = datetime.now()

        # ---- 风控字段 (由风控师注入) ----
        self.tp_levels: list = []         # [(价格, 平仓比例%)]
        self.sl_price: Optional[float] = None
        self.hard_sl: Optional[float] = None
        self.trailing_sl: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.size: float = 0
        self.is_long: bool = (action == "LONG")

        # ---- 追踪止损状态 ----
        self.highest_price: Optional[float] = None   # 做多最高价
        self.lowest_price: Optional[float] = None    # 做空最低价
        self.trail_activated: bool = False
        self.breakeven_activated: bool = False
        self.trail_distance: float = 0
        self.trail_trigger: float = 0
        self.breakeven_trigger: float = 0
        self.atr: float = 0

        # ---- 阶梯止盈状态 ----
        self.tp_filled: List[int] = []  # 已触发阶梯索引

    def inject_risk_params(self, risk_result: dict):
        """从风控师结果注入止损止盈参数"""
        stops = risk_result.get("stops", {})
        self.sl_price = stops.get("final_stop_loss", {}).get("price")
        self.hard_sl = stops.get("hard_stop", {}).get("price")
        self.take_profit = stops.get("take_profit", {}).get("price")
        self.trailing_sl = self.sl_price

        # 阶梯止盈
        self.tp_levels = []
        for step in stops.get("ladder_tp", []):
            self.tp_levels.append((step["target_price"], step.get("close_ratio", "30%")))

        # 追踪参数
        ts = stops.get("trailing_stop", {})
        self.trail_distance = ts.get("trail_distance", 0)
        self.trail_trigger = ts.get("trigger_distance", 0)
        bs = stops.get("breakeven_stop", {})
        self.breakeven_trigger = bs.get("trigger_distance", 0)

        self.size = risk_result.get("size", 0)
        metrics = risk_result.get("risk_metrics", {})
        self.atr = metrics.get("atr", 0)

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
            "hard_sl": self.hard_sl,
            "take_profit": self.take_profit,
            "trailing_sl": self.trailing_sl,
            "size": self.size,
            "trail_activated": self.trail_activated,
            "breakeven_activated": self.breakeven_activated,
            "tp_filled": self.tp_filled,
        }


# ---------------------------------------------------------------------------
# 快循环事件 (执行层回调用)
# ---------------------------------------------------------------------------

class FastLoopEvent:
    """快循环检查触发的事件"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    LADDER_TP = "ladder_tp"
    TRAILING_SL = "trailing_sl_update"
    HARD_STOP = "hard_stop"
    BREAKEVEN = "breakeven"


# ---------------------------------------------------------------------------
# 双循环调度器 (完整实现)
# ---------------------------------------------------------------------------

class DualLoopScheduler:
    """快慢双循环调度器 — Fast Loop 完整实现止损止盈监控"""

    def __init__(self, slow_interval: int = 300, fast_interval: int = 5):
        self.slow_interval = slow_interval
        self.fast_interval = fast_interval
        self.active_plans: Dict[str, TradePlan] = {}
        self._running = False
        self._slow_thread: Optional[threading.Thread] = None
        self._fast_thread: Optional[threading.Thread] = None
        self._slow_callback: Optional[Callable] = None
        self._on_signal: Optional[Callable] = None
        self._on_execution: Optional[Callable] = None  # 执行回调(对接Freqtrade等)
        self._price_fetcher: Optional[Callable] = None  # 外部价格获取器
        self._execution_log: list = []  # 执行日志

    def set_slow_callback(self, callback: Callable):
        """设置慢循环回调 (数据→指标→AI推理→投票)"""
        self._slow_callback = callback

    def set_signal_callback(self, callback: Callable):
        """设置信号回调 (当AI推理完成后触发)"""
        self._on_signal = callback

    def set_execution_callback(self, callback: Callable):
        """
        设置执行回调 — Fast Loop 触发止损止盈时调用
        
        callback(market, symbol, event_type, plan_dict, price) → None
        对接 Freqtrade / MT5 / 手动执行
        """
        self._on_execution = callback

    def set_price_fetcher(self, fetcher: Callable):
        """
        设置实时价格获取器 — Fast Loop 每轮调用
        
        fetcher(symbol) → float (当前价格)
        """
        self._price_fetcher = fetcher

    def start(self):
        """启动双循环"""
        if self._running:
            return
        self._running = True

        self._slow_thread = threading.Thread(target=self._slow_loop, daemon=True)
        self._slow_thread.start()

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
        """快循环: 实时止盈止损监控 + 追踪止损更新"""
        logger.info("快循环已启动 (止损止盈监控)")
        while self._running:
            try:
                if self.active_plans:
                    self._check_active_plans()
            except Exception as e:
                logger.error(f"快循环异常: {e}", exc_info=True)
            time.sleep(self.fast_interval)

    def submit_plan(self, plan: TradePlan):
        """提交交易计划到快循环监控"""
        key = f"{plan.market}:{plan.symbol}"
        # 初始化追踪参考价
        if plan.is_long:
            plan.highest_price = plan.entry_price
        else:
            plan.lowest_price = plan.entry_price
        self.active_plans[key] = plan
        logger.info(
            f"计划已提交: {key} {plan.action} @{plan.entry_price} "
            f"SL={plan.sl_price} TP={plan.take_profit} "
            f"阶梯止盈={len(plan.tp_levels)}级"
        )

    def remove_plan(self, key: str):
        """移除已完成计划"""
        if key in self.active_plans:
            del self.active_plans[key]
            logger.info(f"计划已移除: {key}")

    # ------------------------------------------------------------------
    # 快循环核心: 检查所有活跃计划
    # ------------------------------------------------------------------

    def _check_active_plans(self):
        """检查所有活跃计划的止盈止损状态 — 完整实现"""
        to_remove = []
        events = []

        for key, plan in list(self.active_plans.items()):
            # 获取当前价格
            price = self._get_current_price(plan.symbol)
            if price is None:
                continue

            # 更新追踪最高/最低价
            if plan.is_long:
                plan.highest_price = max(plan.highest_price or price, price)
            else:
                plan.lowest_price = min(plan.lowest_price or price, price)

            # ---- 检查1: 硬止损 (最优先) ----
            event = self._check_hard_stop(plan, price)
            if event:
                events.append(event)
                to_remove.append(key)
                continue

            # ---- 检查2: 初始/最终止损 ----
            event = self._check_stop_loss(plan, price)
            if event:
                events.append(event)
                to_remove.append(key)
                continue

            # ---- 检查3: 阶梯止盈 (分批平仓) ----
            events_ladder, to_remove_plan = self._check_ladder_tp(plan, price)
            events.extend(events_ladder)
            if to_remove_plan:
                to_remove.append(key)
                continue

            # ---- 检查4: 目标止盈 ----
            event = self._check_take_profit(plan, price)
            if event:
                events.append(event)
                to_remove.append(key)
                continue

            # ---- 检查5: 追踪止损更新 (不触发平仓，仅更新SL) ----
            sl_event = self._update_trailing_stop(plan, price)
            if sl_event:
                events.append(sl_event)

        # 清理已触发的计划
        for key in to_remove:
            self.remove_plan(key)

        # 触发执行回调
        for event in events:
            self._fire_execution(event)

    # ------------------------------------------------------------------
    # 各项检查逻辑
    # ------------------------------------------------------------------

    def _check_hard_stop(self, plan: TradePlan, price: float):
        """硬止损检查 — 触发立即平仓"""
        if plan.hard_sl is None:
            return None
        triggered = False
        if plan.is_long and price <= plan.hard_sl:
            triggered = True
        elif not plan.is_long and price >= plan.hard_sl:
            triggered = True

        if triggered:
            logger.warning(
                f"⛔ 硬止损触发! {plan.symbol} 当前价{price} "
                f"硬止损{plan.hard_sl} 入场{plan.entry_price}"
            )
            return {
                "type": FastLoopEvent.HARD_STOP,
                "plan": plan.to_dict(),
                "price": price,
                "message": f"硬止损触发: {plan.symbol} @{price}",
            }
        return None

    def _check_stop_loss(self, plan: TradePlan, price: float):
        """初始止损检查"""
        sl = plan.sl_price
        if sl is None:
            return None
        triggered = False
        if plan.is_long and price <= sl:
            triggered = True
        elif not plan.is_long and price >= sl:
            triggered = True

        if triggered:
            pnl = abs(price - plan.entry_price) * plan.size
            logger.info(
                f"🛑 止损触发! {plan.symbol} 当前价{price} "
                f"止损{sl} 亏损≈${pnl:.0f}"
            )
            return {
                "type": FastLoopEvent.STOP_LOSS,
                "plan": plan.to_dict(),
                "price": price,
                "message": f"止损触发: {plan.symbol} @{price} 亏损${pnl:.0f}",
            }
        return None

    def _check_ladder_tp(self, plan: TradePlan, price: float):
        """
        阶梯止盈检查 — 分批平仓
        
        Returns: (events, should_remove_plan)
        """
        events = []
        all_filled = True

        for i, (tp_price, close_ratio) in enumerate(plan.tp_levels):
            if i in plan.tp_filled:
                continue
            triggered = False
            if plan.is_long and price >= tp_price:
                triggered = True
            elif not plan.is_long and price <= tp_price:
                triggered = True

            if triggered:
                plan.tp_filled.append(i)
                logger.info(
                    f"📊 阶梯止盈 L{i+1} 触发! {plan.symbol} "
                    f"@{tp_price} 平仓{close_ratio}"
                )
                events.append({
                    "type": FastLoopEvent.LADDER_TP,
                    "plan": plan.to_dict(),
                    "price": price,
                    "ladder_index": i + 1,
                    "close_ratio": close_ratio,
                    "target_price": tp_price,
                    "message": f"阶梯止盈L{i+1}: {plan.symbol} @{price} 平{close_ratio}",
                })
            else:
                all_filled = False

        return events, all_filled and len(plan.tp_filled) == len(plan.tp_levels)

    def _check_take_profit(self, plan: TradePlan, price: float):
        """目标止盈检查"""
        tp = plan.take_profit
        if tp is None:
            return None
        triggered = False
        if plan.is_long and price >= tp:
            triggered = True
        elif not plan.is_long and price <= tp:
            triggered = True

        if triggered:
            pnl = abs(price - plan.entry_price) * plan.size
            logger.info(
                f"🎯 止盈触发! {plan.symbol} 当前价{price} "
                f"目标{tp} 盈利≈${pnl:.0f}"
            )
            return {
                "type": FastLoopEvent.TAKE_PROFIT,
                "plan": plan.to_dict(),
                "price": price,
                "message": f"止盈触发: {plan.symbol} @{price} 盈利${pnl:.0f}",
            }
        return None

    def _update_trailing_stop(self, plan: TradePlan, price: float):
        """
        追踪止损动态更新
        
        逻辑：
        1. 盈利超过触发阈值 → 启动追踪
        2. 价格继续朝有利方向移动 → 上移/下移 SL
        3. 盈利超过保本阈值 → SL 移至入场价
        """
        if not plan.trail_distance:
            return None

        changed = False
        old_sl = plan.trailing_sl

        if plan.is_long:
            profit = price - plan.entry_price
            # 保本止损
            if not plan.breakeven_activated and profit > plan.breakeven_trigger:
                plan.trailing_sl = plan.entry_price
                plan.breakeven_activated = True
                changed = True
            # 追踪止损
            elif plan.trail_activated or profit > plan.trail_trigger:
                plan.trail_activated = True
                new_sl = (plan.highest_price or price) - plan.trail_distance
                if new_sl > (plan.trailing_sl or 0):
                    plan.trailing_sl = round(new_sl, 2)
                    changed = True
        else:
            profit = plan.entry_price - price
            # 保本止损
            if not plan.breakeven_activated and profit > plan.breakeven_trigger:
                plan.trailing_sl = plan.entry_price
                plan.breakeven_activated = True
                changed = True
            # 追踪止损
            elif plan.trail_activated or profit > plan.trail_trigger:
                plan.trail_activated = True
                new_sl = (plan.lowest_price or price) + plan.trail_distance
                if new_sl < (plan.trailing_sl or float("inf")):
                    plan.trailing_sl = round(new_sl, 2)
                    changed = True

        if changed:
            logger.info(
                f"🔄 追踪止损更新: {plan.symbol} "
                f"SL {old_sl} → {plan.trailing_sl} (当前价{price})"
            )
            return {
                "type": FastLoopEvent.TRAILING_SL,
                "plan": plan.to_dict(),
                "price": price,
                "old_sl": old_sl,
                "new_sl": plan.trailing_sl,
                "message": f"追踪止损: {plan.symbol} SL {old_sl}→{plan.trailing_sl}",
            }

        return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格 (通过外部注入的价格获取器)"""
        if self._price_fetcher:
            try:
                return self._price_fetcher(symbol)
            except Exception as e:
                logger.error(f"价格获取失败 [{symbol}]: {e}")
                return None
        return None

    def _fire_execution(self, event: dict):
        """触发执行回调"""
        self._execution_log.append({
            "timestamp": datetime.now().isoformat(),
            **event,
        })
        if self._on_execution:
            try:
                self._on_execution(event)
            except Exception as e:
                logger.error(f"执行回调异常: {e}")

    def get_execution_log(self, limit: int = 50) -> list:
        """获取执行日志"""
        return self._execution_log[-limit:]


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
