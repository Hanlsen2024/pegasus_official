"""
回测引擎 — 基于历史数据的策略验证

功能:
- 逐K线遍历(无未来函数)
- 规则化信号生成(对标AI Agent逻辑，无LLM开销)
- 完整风控模拟(阶梯止盈/追踪止损/保本/硬止损)
- 绩效报告（收益率/胜率/盈亏比/夏普/最大回撤/权益曲线）
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("picas.backtester")


# =========================================================================
# 数据结构
# =========================================================================

@dataclass
class Trade:
    """单笔交易记录"""
    entry_time: str = ""
    exit_time: str = ""
    symbol: str = ""
    direction: str = "LONG"       # LONG / SHORT
    entry_price: float = 0
    exit_price: float = 0
    size: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    exit_reason: str = ""         # take_profit / stop_loss / hard_stop / ladder_tp / end_of_data
    ladder_fills: List[int] = field(default_factory=list)  # 已触发的阶梯索引
    high_price: float = 0
    low_price: float = 0


@dataclass
class BacktestResult:
    """回测结果"""
    market: str = ""
    symbol: str = ""
    start_date: str = ""
    end_date: str = ""
    total_bars: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0
    total_pnl: float = 0
    total_pnl_pct: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    profit_factor: float = 0
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0
    annual_return_pct: float = 0
    trades: List[dict] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


# =========================================================================
# 规则化信号生成 (对标 AI Agent 判断框架，无 LLM 调用)
# =========================================================================

class RuleSignalGenerator:
    """
    纯规则信号引擎 — 用技术指标生成 BUY/SELL/HOLD + 得分

    对标 3 个 AI Agent 的逻辑:
    - Indicator Agent:  EMA交叉 + RSI + MACD 
    - Pattern Agent:   布林带位置 + K线形态
    - Trend Agent:     价格结构 + ADX方向
    """

    @staticmethod
    def generate(indicators: dict, price_summary: str = "") -> dict:
        """
        生成规则化交易信号

        Returns:
            {"signal": "BUY"/"SELL"/"HOLD", "score": float,
             "buy_score": float, "sell_score": float, "signals": list}
        """
        trend = indicators.get("trend", {})
        momentum = indicators.get("momentum", {})
        volatility = indicators.get("volatility", {})
        sr = indicators.get("sr_levels", {})

        buy_score = 0.0
        sell_score = 0.0
        signals = []

        # ---- 1. EMA 交叉 (权重 30) ----
        ema_cross = trend.get("ema_cross", "")
        if ema_cross == "bullish":
            buy_score += 30
            signals.append("EMA金叉")
        elif ema_cross == "bearish":
            sell_score += 30
            signals.append("EMA死叉")

        # ---- 2. 价格 vs EMA50 (权重 20) ----
        price_vs_ema = trend.get("price_vs_ema50", "")
        if price_vs_ema == "above":
            buy_score += 20
            signals.append("价格>EMA50")
        elif price_vs_ema == "below":
            sell_score += 20
            signals.append("价格<EMA50")

        # ---- 3. MACD 方向 (权重 25) ----
        macd_dir = trend.get("macd", {}).get("direction", "")
        if macd_dir == "up":
            buy_score += 25
            signals.append("MACD转多")
        elif macd_dir == "down":
            sell_score += 25
            signals.append("MACD转空")

        # ---- 4. RSI 区域 (权重 25) ----
        rsi_zone = momentum.get("rsi_zone", "")
        rsi_val = float(momentum.get("rsi14", 50))
        if rsi_zone == "oversold":
            buy_score += 25
            signals.append("RSI超卖")
        elif rsi_zone == "overbought":
            sell_score += 25
            signals.append("RSI超买")
        elif rsi_val < 40:
            buy_score += 8
        elif rsi_val > 60:
            sell_score += 8

        # ---- 5. 布林带位置 (权重 20) ----
        bb_pos = volatility.get("bollinger", {}).get("position", "")
        if bb_pos == "below_lower":
            buy_score += 20
            signals.append("布林下轨支撑")
        elif bb_pos == "above_upper":
            sell_score += 20
            signals.append("布林上轨压力")

        # ---- 6. 随机指标 (权重 15) ----
        stoch_k = float(momentum.get("stoch_k", 50))
        if stoch_k < 20:
            buy_score += 15
            signals.append("KD超卖")
        elif stoch_k > 80:
            sell_score += 15
            signals.append("KD超买")

        # ---- 7. ROC 变动率 (权重 10) ----
        roc = float(momentum.get("roc10", 0))
        if roc > 2:
            buy_score += 10
        elif roc < -2:
            sell_score += 10

        # 归一化到 [-1, 1]
        max_possible = 145  # 总分
        buy_score /= max_possible
        sell_score /= max_possible
        net = buy_score - sell_score

        # 决定信号
        if net > 0.15:
            signal = "BUY"
        elif net < -0.15:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "score": round(net, 4),
            "buy_score": round(buy_score, 4),
            "sell_score": round(sell_score, 4),
            "signals": signals,
        }


# =========================================================================
# 回测引擎
# =========================================================================

class BacktestEngine:
    """历史回测引擎 — 逐K线模拟交易"""

    def __init__(self, market: str = "gold", symbol: str = "",
                 initial_capital: float = 10000, risk_per_trade: float = 0.02,
                 commission: float = 0.001):
        self.market = market
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission  # 双边手续费

        # ---- 风控参数 (从 config 读取) ----
        self._load_risk_params()

    def _load_risk_params(self):
        """加载风控参数"""
        from config.loader import get_risk_config
        if self.market == "crypto":
            cfg = get_risk_config("crypto")
        else:
            cfg = get_risk_config()

        self.default_tp_pct = cfg.get("default_tp_pct", 5.0) / 100
        self.default_sl_pct = cfg.get("default_sl_pct", 3.0) / 100
        self.atr_mult = cfg.get("atr_trailing_mult", 2.0)
        self.ladder_tp = cfg.get("ladder_tp", [])
        self.hard_stop_pct = cfg.get("hard_stop_pct", 0.08)

        # crypto 专属
        if self.market == "crypto":
            self.trail_trigger_atr = cfg.get("trail_trigger_atr", 2.0)
            self.trail_distance_atr = cfg.get("trail_distance_atr", 1.5)
            self.breakeven_trigger_atr = cfg.get("breakeven_trigger_atr", 1.5)
        else:
            self.trail_trigger_atr = 2.0
            self.trail_distance_atr = 1.5
            self.breakeven_trigger_atr = 1.0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        执行回测

        Args:
            df: 历史K线数据 (columns: open, high, low, close, volume)

        Returns:
            BacktestResult: 完整回测报告
        """
        if df.empty or len(df) < 50:
            logger.error("回测数据不足(最少50根K线)")
            return BacktestResult(market=self.market, symbol=self.symbol)

        from data.indicator_bank import compute_indicator_series, snap_indicators_at

        # ---- 一次计算全部指标序列 (O(n)) ----
        indicator_series = compute_indicator_series(df)

        capital = self.initial_capital
        open_trade: Optional[Trade] = None
        trades: List[Trade] = []
        equity_curve: List[dict] = []

        min_bars = 30  # 指标预热

        for i in range(min_bars, len(df)):
            current_bar = df.iloc[i]
            bar_time = str(current_bar.name) if hasattr(current_bar, 'name') else str(i)
            current_price = float(current_bar["close"])
            current_high = float(current_bar["high"])
            current_low = float(current_bar["low"])

            # ---- 处理持仓 ----
            if open_trade is not None:
                # 更新最高/最低价
                if open_trade.direction == "LONG":
                    open_trade.high_price = max(open_trade.high_price, current_high)
                else:
                    open_trade.low_price = min(open_trade.low_price, current_low)

                # 检查退出信号 (返回: (exit_price, reason, ladder_idx) 或 None)
                exit_info = self._check_exit(open_trade, current_bar)
                if exit_info:
                    exit_price, exit_reason, ladder_idx = exit_info

                    if exit_reason in ("hard_stop", "stop_loss", "take_profit", "ladder_tp_complete", "end_of_data"):
                        # ---- 完全平仓 ----
                        open_trade.exit_price = exit_price
                        open_trade.exit_time = bar_time
                        open_trade.exit_reason = exit_reason

                        if open_trade.direction == "LONG":
                            raw_pnl = (exit_price - open_trade.entry_price) * open_trade.size
                        else:
                            raw_pnl = (open_trade.entry_price - exit_price) * open_trade.size
                        fee = open_trade.entry_price * open_trade.size * self.commission * 2
                        open_trade.pnl = raw_pnl - fee
                        open_trade.pnl_pct = (open_trade.pnl / capital) * 100
                        capital += open_trade.pnl

                        if exit_reason == "ladder_tp_complete" and ladder_idx is not None:
                            open_trade.ladder_fills.append(ladder_idx)

                        trades.append(open_trade)
                        open_trade = None

                    elif exit_reason == "ladder_tp":
                        # ---- 部分平仓 (阶梯止盈) ----
                        step = self.ladder_tp[ladder_idx]
                        close_ratio = step.get("close_ratio", 0.3)
                        orig_size = getattr(open_trade, '_orig_size', open_trade.size)
                        partial_size = orig_size * close_ratio

                        if open_trade.direction == "LONG":
                            raw_pnl = (exit_price - open_trade.entry_price) * partial_size
                        else:
                            raw_pnl = (open_trade.entry_price - exit_price) * partial_size
                        fee = open_trade.entry_price * partial_size * self.commission * 2
                        partial_pnl = raw_pnl - fee

                        # 记录部分平仓
                        part_trade = Trade(
                            entry_time=open_trade.entry_time,
                            exit_time=bar_time,
                            symbol=open_trade.symbol,
                            direction=open_trade.direction,
                            entry_price=open_trade.entry_price,
                            exit_price=exit_price,
                            size=round(partial_size, 6),
                            pnl=round(partial_pnl, 2),
                            pnl_pct=round((partial_pnl / capital) * 100, 4),
                            exit_reason="ladder_tp",
                            ladder_fills=[ladder_idx],
                        )
                        trades.append(part_trade)
                        capital += partial_pnl

                        # 缩减剩余仓位
                        open_trade.size -= partial_size
                        open_trade.ladder_fills.append(ladder_idx)

                        # 阶梯止盈后保本
                        open_trade._sl_price = open_trade.entry_price
                        open_trade._breakeven_activated = True

                        if open_trade.size <= 0:
                            open_trade = None

            # ---- 从预计算序列中提取当前指标快照 (O(1)) ----
            indicators = snap_indicators_at(indicator_series, i, current_price)
            if "error" in indicators:
                continue

            signal_result = RuleSignalGenerator.generate(indicators)

            # ---- 开仓 ----
            if open_trade is None and signal_result["signal"] in ("BUY", "SELL"):
                direction = "LONG" if signal_result["signal"] == "BUY" else "SHORT"
                entry_price = current_price

                # 计算头寸大小
                atr = float(indicators.get("volatility", {}).get("atr14", 0))
                if atr <= 0:
                    atr = entry_price * 0.01

                # 止损 = ATR × 倍数
                sl_distance = atr * self.atr_mult
                if direction == "LONG":
                    sl_price = round(entry_price - sl_distance, 4)
                    tp_price = round(entry_price + sl_distance * 2.0, 4)  # 默认 R:R=2
                else:
                    sl_price = round(entry_price + sl_distance, 4)
                    tp_price = round(entry_price - sl_distance * 2.0, 4)

                # 硬止损
                hard_sl_pct = getattr(self, 'hard_stop_pct', 0.08)
                if direction == "LONG":
                    hard_sl = round(entry_price * (1 - hard_sl_pct), 4)
                else:
                    hard_sl = round(entry_price * (1 + hard_sl_pct), 4)

                # 风险金额 → 头寸大小
                risk_amount = capital * self.risk_per_trade
                sl_pct = sl_distance / entry_price
                if sl_pct > 0:
                    size = risk_amount / (entry_price * sl_pct)
                else:
                    size = 0

                if size <= 0:
                    continue

                trade = Trade(
                    entry_time=bar_time,
                    symbol=self.symbol or self.market,
                    direction=direction,
                    entry_price=entry_price,
                    size=round(size, 6),
                    high_price=entry_price,
                    low_price=entry_price,
                )
                # 附加风控参数
                trade._sl_price = sl_price
                trade._tp_price = tp_price
                trade._hard_sl = hard_sl
                trade._atr = atr
                trade._sl_distance = sl_distance
                trade._orig_size = size
                trade._trail_activated = False
                trade._breakeven_activated = False
                trade._ladder_triggered = set()

                open_trade = trade

            # ---- 记录权益曲线 ----
            equity_curve.append({
                "time": bar_time,
                "equity": round(capital, 2),
                "price": round(current_price, 2),
                "has_position": open_trade is not None,
            })

        # ---- 强制平仓最后持仓 ----
        if open_trade is not None:
            last_bar = df.iloc[-1]
            last_price = float(last_bar["close"])
            open_trade.exit_price = last_price
            open_trade.exit_time = str(last_bar.name) if hasattr(last_bar, 'name') else str(len(df) - 1)
            open_trade.exit_reason = "end_of_data"

            if open_trade.direction == "LONG":
                raw_pnl = (last_price - open_trade.entry_price) * open_trade.size
            else:
                raw_pnl = (open_trade.entry_price - last_price) * open_trade.size
            fee = open_trade.entry_price * open_trade.size * self.commission * 2
            open_trade.pnl = raw_pnl - fee
            open_trade.pnl_pct = (open_trade.pnl / capital) * 100
            capital += open_trade.pnl
            trades.append(open_trade)
            open_trade = None

            equity_curve.append({
                "time": open_trade.exit_time,
                "equity": round(capital, 2),
                "price": round(last_price, 2),
                "has_position": False,
            })

        # ---- 构建结果 ----
        return self._build_result(df, trades, equity_curve)

    # ------------------------------------------------------------------
    # 退出检查
    # ------------------------------------------------------------------

    def _check_exit(self, trade: Trade,
                    bar: pd.Series) -> Optional[Tuple[float, str, Optional[int]]]:
        """
        检查是否触发退出条件
        优先级: 硬止损 > 初始止损 > 阶梯止盈 > 目标止盈

        Returns:
            (exit_price, reason, ladder_index) 或 None
            reason: hard_stop/stop_loss/ladder_tp/ladder_tp_complete/take_profit
            ladder_tp 表示部分平仓,ladder_tp_complete 表示阶梯全部执行完毕
        """
        price = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        is_long = trade.direction == "LONG"

        # ---- 1. 硬止损 ----
        hs = getattr(trade, '_hard_sl', None)
        if hs:
            if is_long and low <= hs:
                return (hs, "hard_stop", None)
            elif not is_long and high >= hs:
                return (hs, "hard_stop", None)

        # ---- 2. 初始/最终止损 (含追踪) ----
        sl = getattr(trade, '_sl_price', None)
        if sl:
            if is_long and low <= sl:
                return (sl, "stop_loss", None)
            elif not is_long and high >= sl:
                return (sl, "stop_loss", None)

        # ---- 3. 阶梯止盈 ----
        ladder = self.ladder_tp
        triggered = getattr(trade, '_ladder_triggered', set())
        if not ladder:
            pass  # 无阶梯配置则跳过
        else:
            for idx, step in enumerate(ladder):
                if idx in triggered:
                    continue
                tp_pct = step.get("pct", 5) / 100
                close_ratio = step.get("close_ratio", 0.3)
                if is_long:
                    tp_price = trade.entry_price * (1 + tp_pct)
                    if high >= tp_price:
                        triggered.add(idx)
                        trade._ladder_triggered = triggered
                        # 检查是否最后一阶
                        total_closed = sum(
                            ladder[i].get("close_ratio", 0) for i in triggered
                        )
                        remaining = 1.0 - total_closed
                        if remaining <= 0.0001:
                            return (tp_price, "ladder_tp_complete", idx)
                        return (tp_price, "ladder_tp", idx)
                else:
                    tp_price = trade.entry_price * (1 - tp_pct)
                    if low <= tp_price:
                        triggered.add(idx)
                        trade._ladder_triggered = triggered
                        total_closed = sum(
                            ladder[i].get("close_ratio", 0) for i in triggered
                        )
                        remaining = 1.0 - total_closed
                        if remaining <= 0.0001:
                            return (tp_price, "ladder_tp_complete", idx)
                        return (tp_price, "ladder_tp", idx)

        # ---- 4. 目标止盈 ----
        tp = getattr(trade, '_tp_price', None)
        if tp:
            if is_long and high >= tp:
                return (tp, "take_profit", None)
            elif not is_long and low <= tp:
                return (tp, "take_profit", None)

        # ---- 5. 追踪止损更新 (不触发退出，仅更新SL) ----
        self._update_trailing_stop(trade, bar)

        return None

    def _update_trailing_stop(self, trade: Trade, bar: pd.Series):
        """更新追踪止损价位"""
        price = float(bar["close"])
        is_long = trade.direction == "LONG"
        atr = getattr(trade, '_atr', 0)
        if atr <= 0:
            return

        trail_trigger = atr * self.trail_trigger_atr
        trail_dist = atr * self.trail_distance_atr
        breakeven_trigger = atr * self.breakeven_trigger_atr

        if is_long:
            profit = price - trade.entry_price
            # 保本
            if not getattr(trade, '_breakeven_activated', False) and profit > breakeven_trigger:
                trade._sl_price = trade.entry_price
                trade._breakeven_activated = True
            # 追踪
            elif getattr(trade, '_trail_activated', False) or profit > trail_trigger:
                trade._trail_activated = True
                new_sl = trade.high_price - trail_dist
                if new_sl > getattr(trade, '_sl_price', 0):
                    trade._sl_price = round(new_sl, 4)
        else:
            profit = trade.entry_price - price
            # 保本
            if not getattr(trade, '_breakeven_activated', False) and profit > breakeven_trigger:
                trade._sl_price = trade.entry_price
                trade._breakeven_activated = True
            # 追踪
            elif getattr(trade, '_trail_activated', False) or profit > trail_trigger:
                trade._trail_activated = True
                new_sl = trade.low_price + trail_dist
                if new_sl < getattr(trade, '_sl_price', float("inf")):
                    trade._sl_price = round(new_sl, 4)

    # ------------------------------------------------------------------
    # 绩效报告
    # ------------------------------------------------------------------

    def _build_result(self, df: pd.DataFrame, trades: List[Trade],
                      equity_curve: List[dict]) -> BacktestResult:
        """构建回测绩效报告"""
        start_time = str(df.index[0])
        end_time = str(df.index[-1])

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        win_count = len(winning)
        lose_count = len(losing)
        trade_count = len(trades)
        win_rate = win_count / trade_count if trade_count > 0 else 0

        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t.pnl for t in losing])) if losing else 0
        total_win = sum(t.pnl for t in winning)
        total_loss = abs(sum(t.pnl for t in losing))
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf") if total_win > 0 else 0

        # 最大回撤
        max_dd = 0
        peak = self.initial_capital
        for point in equity_curve:
            eq = point["equity"]
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # 夏普比率 (简化: 年化)
        if len(equity_curve) >= 2:
            returns = []
            prev = equity_curve[0]["equity"]
            for pt in equity_curve[1:]:
                r = (pt["equity"] - prev) / prev if prev > 0 else 0
                returns.append(r)
                prev = pt["equity"]
            if returns:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # 年化收益
        if start_time and end_time:
            try:
                start_dt = pd.to_datetime(start_time)
                end_dt = pd.to_datetime(end_time)
                days = (end_dt - start_dt).days
                if days > 0:
                    annual_return = (1 + total_pnl_pct / 100) ** (365 / days) - 1
                    annual_return_pct = annual_return * 100
                else:
                    annual_return_pct = total_pnl_pct
            except Exception:
                annual_return_pct = total_pnl_pct
        else:
            annual_return_pct = total_pnl_pct

        # 交易明细
        trade_list = []
        for t in trades:
            trade_list.append({
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 4),
                "exit_reason": t.exit_reason,
            })

        result = BacktestResult(
            market=self.market,
            symbol=self.symbol or self.market,
            start_date=start_time,
            end_date=end_time,
            total_bars=len(df),
            total_trades=trade_count,
            winning_trades=win_count,
            losing_trades=lose_count,
            win_rate=round(win_rate * 100, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 999,
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            annual_return_pct=round(annual_return_pct, 2),
            trades=trade_list,
            equity_curve=equity_curve,
        )

        return result


# =========================================================================
# 便捷函数
# =========================================================================

def run_backtest(market: str = "gold", symbol: str = None,
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 lookback_days: int = 180) -> dict:
    """
    一键运行回测

    Args:
        market: 市场类型
        symbol: 标的代码
        initial_capital: 初始资金
        risk_per_trade: 单笔风险比例
        lookback_days: 回看天数

    Returns:
        dict: 回测报告 (可直接 JSON 序列化)
    """
    from data.fetcher import get_market_data, get_crypto_data

    logger.info(f"回测启动: market={market} symbol={symbol} capital={initial_capital}")

    # 获取历史数据
    if market == "crypto" and symbol:
        df = get_crypto_data(symbol, interval="4h")
    else:
        df = get_market_data(market=market, symbol=symbol)

    if df is None or df.empty:
        return {"error": f"无法获取 {market}/{symbol} 历史数据"}

    # 限制回看范围
    if lookback_days > 0:
        cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
        df = df[df.index >= cutoff]

    if len(df) < 50:
        return {"error": f"数据不足(仅{len(df)}根K线，需要≥50)"}

    # 运行回测
    engine = BacktestEngine(
        market=market,
        symbol=symbol or market,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
    )
    result = engine.run(df)

    # 转为可序列化字典
    return {
        "market": result.market,
        "symbol": result.symbol,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "total_bars": result.total_bars,
        "initial_capital": initial_capital,
        "final_capital": round(initial_capital + result.total_pnl, 2),
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "total_pnl_pct": result.total_pnl_pct,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "annual_return_pct": result.annual_return_pct,
        "trades": result.trades[-20:],  # 最近20笔
        "equity_curve": result.equity_curve[::max(1, len(result.equity_curve) // 100)],  # 采样
    }


__all__ = ["BacktestEngine", "BacktestResult", "RuleSignalGenerator", "run_backtest"]
