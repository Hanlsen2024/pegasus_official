"""
多分析师模块 - 每种策略为一个独立的"分析师"，各自给出交易信号
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 分析师抽象基类
# ---------------------------------------------------------------------------

class BaseAnalyst(ABC):
    """所有分析师的基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def analyze(self, market_data: dict) -> dict:
        """
        分析市场数据并返回信号

        Args:
            market_data: {"df": pd.DataFrame} 包含 OHLCV 数据

        Returns:
            {"action": "BUY"|"SELL"|"HOLD", "score": float, "reason": str}
        """
        ...

    def __repr__(self):
        return f"{self.name}(w={self.weight})"


# ---------------------------------------------------------------------------
# 1. MA 双均线交叉分析师
# ---------------------------------------------------------------------------

class MACrossoverAnalyst(BaseAnalyst):
    """MA20 / MA50 双均线交叉策略"""

    def __init__(self, short_ma: int = 20, long_ma: int = 50, weight: float = 1.2):
        super().__init__(name=f"MA{short_ma}/{long_ma}", weight=weight)
        self.short_ma = short_ma
        self.long_ma = long_ma

    def analyze(self, market_data: dict) -> dict:
        df = market_data["df"].copy()
        if len(df) < self.long_ma:
            return {"action": "HOLD", "score": 0.0,
                    "reason": f"数据不足(需>{self.long_ma}条)"}

        df["ma_short"] = df["close"].rolling(self.short_ma).mean()
        df["ma_long"] = df["close"].rolling(self.long_ma).mean()

        current = df["ma_short"].iloc[-1]
        current_long = df["ma_long"].iloc[-1]
        prev = df["ma_short"].iloc[-2]
        prev_long = df["ma_long"].iloc[-2]

        # 金叉：短线上穿长线
        if pd.notna(current) and pd.notna(current_long):
            if prev <= prev_long and current > current_long:
                return {"action": "BUY", "score": 0.75,
                        "reason": f"金叉 MA{self.short_ma}↑MA{self.long_ma}"}
            elif prev >= prev_long and current < current_long:
                return {"action": "SELL", "score": 0.75,
                        "reason": f"死叉 MA{self.short_ma}↓MA{self.long_ma}"}
            elif current > current_long:
                return {"action": "BUY", "score": 0.35,
                        "reason": f"多头排列 MA{self.short_ma}>{self.long_ma}"}
            else:
                return {"action": "SELL", "score": 0.35,
                        "reason": f"空头排列 MA{self.short_ma}<{self.long_ma}"}

        return {"action": "HOLD", "score": 0.0, "reason": "均线数据无效"}


# ---------------------------------------------------------------------------
# 2. RSI 超买超卖分析师
# ---------------------------------------------------------------------------

class RSIAnalyst(BaseAnalyst):
    """RSI 指标 - 70/30 超买超卖"""

    def __init__(self, period: int = 14, overbought: int = 70,
                 oversold: int = 30, weight: float = 1.0):
        super().__init__(name=f"RSI({period})", weight=weight)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def _calc_rsi(self, df: pd.DataFrame) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def analyze(self, market_data: dict) -> dict:
        df = market_data["df"].copy()
        if len(df) < self.period + 1:
            return {"action": "HOLD", "score": 0.0,
                    "reason": f"数据不足(需>{self.period}条)"}

        rsi = self._calc_rsi(df)
        current = rsi.iloc[-1]
        prev = rsi.iloc[-2] if len(rsi) >= 2 else current

        if pd.isna(current):
            return {"action": "HOLD", "score": 0.0, "reason": "RSI 计算无效"}

        # 超卖反弹
        if current < self.oversold:
            return {"action": "BUY", "score": 0.65,
                    "reason": f"超卖 RSI={current:.1f}<{self.oversold}"}
        # 超买回落
        elif current > self.overbought:
            return {"action": "SELL", "score": 0.65,
                    "reason": f"超买 RSI={current:.1f}>{self.overbought}"}
        # 从低区回升
        elif prev < self.oversold and current >= self.oversold:
            return {"action": "BUY", "score": 0.50,
                    "reason": f"超卖回升 RSI={current:.1f}"}
        # 从高区回落
        elif prev > self.overbought and current <= self.overbought:
            return {"action": "SELL", "score": 0.50,
                    "reason": f"超买回落 RSI={current:.1f}"}
        else:
            return {"action": "HOLD", "score": 0.0,
                    "reason": f"中性 RSI={current:.1f}"}


# ---------------------------------------------------------------------------
# 3. MACD 分析师
# ---------------------------------------------------------------------------

class MACDAnalyst(BaseAnalyst):
    """MACD 指标"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 weight: float = 1.0):
        super().__init__(name=f"MACD({fast},{slow},{signal})", weight=weight)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def analyze(self, market_data: dict) -> dict:
        df = market_data["df"].copy()
        if len(df) < self.slow + self.signal:
            return {"action": "HOLD", "score": 0.0,
                    "reason": f"数据不足(需>{self.slow + self.signal}条)"}

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram = macd_line - signal_line

        curr_macd = macd_line.iloc[-1]
        curr_sig = signal_line.iloc[-1]
        curr_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        if pd.isna(curr_hist):
            return {"action": "HOLD", "score": 0.0, "reason": "MACD 无效"}

        # 金叉
        if prev_hist <= 0 and curr_hist > 0:
            return {"action": "BUY", "score": 0.70,
                    "reason": "MACD 金叉"}
        # 死叉
        elif prev_hist >= 0 and curr_hist < 0:
            return {"action": "SELL", "score": 0.70,
                    "reason": "MACD 死叉"}
        elif curr_hist > 0:
            return {"action": "BUY", "score": 0.30,
                    "reason": "MACD 动量偏多"}
        else:
            return {"action": "SELL", "score": 0.30,
                    "reason": "MACD 动量偏空"}


# ---------------------------------------------------------------------------
# 4. 趋势/通道分析师
# ---------------------------------------------------------------------------

class TrendAnalyst(BaseAnalyst):
    """价格相对均线位置 + 近期涨跌幅综合判断"""

    def __init__(self, ma_period: int = 50, weight: float = 0.8):
        super().__init__(name=f"Trend(MA{ma_period})", weight=weight)
        self.ma_period = ma_period

    def analyze(self, market_data: dict) -> dict:
        df = market_data["df"].copy()
        if len(df) < self.ma_period + 5:
            return {"action": "HOLD", "score": 0.0,
                    "reason": f"数据不足(需>{self.ma_period + 5}条)"}

        df["ma"] = df["close"].rolling(self.ma_period).mean()
        current_price = df["close"].iloc[-1]
        ma_val = df["ma"].iloc[-1]

        if pd.isna(ma_val):
            return {"action": "HOLD", "score": 0.0, "reason": "MA 无效"}

        # 偏离均线幅度
        deviation = (current_price - ma_val) / ma_val

        # 5日动量
        pct_5d = df["close"].pct_change(5).iloc[-1] or 0

        if deviation > 0.05:  # 价格高于均线 5%
            return {"action": "SELL", "score": 0.40,
                    "reason": f"乖离过大 +{deviation:.1%}"}
        elif deviation < -0.05:
            return {"action": "BUY", "score": 0.40,
                    "reason": f"超跌 -{abs(deviation):.1%}"}
        elif pct_5d > 0.02:
            return {"action": "BUY", "score": 0.25,
                    "reason": f"短期强势 +{pct_5d:.1%}"}
        elif pct_5d < -0.02:
            return {"action": "SELL", "score": 0.25,
                    "reason": f"短期弱势 -{abs(pct_5d):.1%}"}
        else:
            return {"action": "HOLD", "score": 0.0, "reason": "趋势平稳"}


# ---------------------------------------------------------------------------
# 5. 交易量/波动率分析师
# ---------------------------------------------------------------------------

class VolumeAnalyst(BaseAnalyst):
    """量价配合分析"""

    def __init__(self, weight: float = 0.6):
        super().__init__(name="Volume", weight=weight)

    def analyze(self, market_data: dict) -> dict:
        df = market_data["df"].copy()
        if "volume" not in df.columns or len(df) < 10:
            return {"action": "HOLD", "score": 0.0,
                    "reason": "无成交量数据"}

        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]
        price_chg = df["close"].pct_change().iloc[-1]

        if pd.isna(avg_vol) or avg_vol == 0:
            return {"action": "HOLD", "score": 0.0, "reason": "成交量无效"}

        vol_ratio = current_vol / avg_vol

        if vol_ratio > 1.5 and price_chg > 0:
            return {"action": "BUY", "score": 0.50,
                    "reason": f"放量上涨 vol×{vol_ratio:.1f}"}
        elif vol_ratio > 1.5 and price_chg < 0:
            return {"action": "SELL", "score": 0.50,
                    "reason": f"放量下跌 vol×{vol_ratio:.1f}"}
        else:
            return {"action": "HOLD", "score": 0.0, "reason": "量价正常"}


# ---------------------------------------------------------------------------
# 获取全部分析师
# ---------------------------------------------------------------------------

def get_all_analysts() -> list:
    """返回所有启用的分析师实例"""
    return [
        MACrossoverAnalyst(short_ma=20, long_ma=50, weight=1.2),
        RSIAnalyst(period=14, overbought=70, oversold=30, weight=1.0),
        MACDAnalyst(fast=12, slow=26, signal=9, weight=1.0),
        TrendAnalyst(ma_period=50, weight=0.8),
        VolumeAnalyst(weight=0.6),
    ]
