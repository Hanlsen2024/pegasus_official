"""
黄金专属分析师 — 4小时周期 + 日线辅助
与股票分析完全独立，专注黄金特有规律
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseGoldAnalyst(ABC):
    """黄金分析师抽象基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = f"🥇{name}"
        self.weight = weight

    @abstractmethod
    def analyze(self, gold_data: dict) -> dict:
        """
        Args:
            gold_data: {"df_4h": DataFrame, "df_daily": DataFrame, "dxy": DataFrame}
        Returns:
            {"action": "BUY"|"SELL"|"HOLD", "score": float, "reason": str}
        """
        ...


# ===================================================================
# 1. 4H EMA 12/26 双均线 — 黄金短期动量
# ===================================================================
class GoldEMA12_26(BaseGoldAnalyst):
    """4H 周期 EMA12/26 交叉 — 黄金日内主力参考"""

    def __init__(self):
        super().__init__(name="4H-EMA(12,26)交叉", weight=1.2)

    def analyze(self, gold_data: dict) -> dict:
        df = gold_data.get("df_4h", pd.DataFrame()).copy()
        if len(df) < 30:
            return {"action": "HOLD", "score": 0.0, "reason": "4H数据不足"}

        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

        short = df["ema12"].iloc[-1]
        long = df["ema26"].iloc[-1]
        prev_s = df["ema12"].iloc[-2]
        prev_l = df["ema26"].iloc[-2]

        if pd.isna(short) or pd.isna(long):
            return {"action": "HOLD", "score": 0.0, "reason": "EMA数据无效"}

        # 金叉
        if prev_s <= prev_l and short > long:
            return {"action": "BUY", "score": 0.70,
                    "reason": "4H金叉 EMA12↑EMA26"}
        # 死叉
        elif prev_s >= prev_l and short < long:
            return {"action": "SELL", "score": 0.70,
                    "reason": "4H死叉 EMA12↓EMA26"}
        elif short > long:
            return {"action": "BUY", "score": 0.30,
                    "reason": "4H多头排列"}
        else:
            return {"action": "SELL", "score": 0.30,
                    "reason": "4H空头排列"}


# ===================================================================
# 2. 4H RSI(14) — 黄金超买超卖
# ===================================================================
class GoldRSI(BaseGoldAnalyst):
    """4H RSI(14) — 黄金对超买超卖敏感度高于股票"""

    def __init__(self):
        super().__init__(name="4H-RSI(14)", weight=1.0)

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def analyze(self, gold_data: dict) -> dict:
        df = gold_data.get("df_4h", pd.DataFrame()).copy()
        if len(df) < 16:
            return {"action": "HOLD", "score": 0.0, "reason": "数据不足"}

        rsi = self._rsi(df["close"], 14)
        val = rsi.iloc[-1]
        prev = rsi.iloc[-2]

        if pd.isna(val):
            return {"action": "HOLD", "score": 0.0, "reason": "RSI无效"}

        # 黄金 RSI 区间更窄：25/75
        if val < 25:
            return {"action": "BUY", "score": 0.70,
                    "reason": f"4H超卖 RSI={val:.1f}<25"}
        elif val > 75:
            return {"action": "SELL", "score": 0.70,
                    "reason": f"4H超买 RSI={val:.1f}>75"}
        elif val < 35 and not pd.isna(prev) and val > prev:
            return {"action": "BUY", "score": 0.40,
                    "reason": f"4H底部回升 RSI={val:.1f}"}
        elif val > 65 and not pd.isna(prev) and val < prev:
            return {"action": "SELL", "score": 0.40,
                    "reason": f"4H顶部回落 RSI={val:.1f}"}
        elif val < 40:
            return {"action": "BUY", "score": 0.20,
                    "reason": f"4H偏多区间 RSI={val:.1f}"}
        elif val > 60:
            return {"action": "SELL", "score": 0.20,
                    "reason": f"4H偏空区间 RSI={val:.1f}"}
        else:
            return {"action": "HOLD", "score": 0.0,
                    "reason": f"4H中性 RSI={val:.1f}"}


# ===================================================================
# 3. 4H MACD — 经典信号
# ===================================================================
class GoldMACD(BaseGoldAnalyst):
    """4H MACD(12,26,9)"""

    def __init__(self):
        super().__init__(name="4H-MACD", weight=1.0)

    def analyze(self, gold_data: dict) -> dict:
        df = gold_data.get("df_4h", pd.DataFrame()).copy()
        if len(df) < 40:
            return {"action": "HOLD", "score": 0.0, "reason": "数据不足"}

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        curr = hist.iloc[-1]
        prev = hist.iloc[-2]

        if pd.isna(curr):
            return {"action": "HOLD", "score": 0.0, "reason": "MACD无效"}

        if prev <= 0 and curr > 0:
            return {"action": "BUY", "score": 0.65, "reason": "4H MACD金叉"}
        elif prev >= 0 and curr < 0:
            return {"action": "SELL", "score": 0.65, "reason": "4H MACD死叉"}
        elif curr > 0:
            return {"action": "BUY", "score": 0.25, "reason": "4H MACD动量偏多"}
        else:
            return {"action": "SELL", "score": 0.25, "reason": "4H MACD动量偏空"}


# ===================================================================
# 4. 4H 布林带(20,2) — 黄金突破/回归
# ===================================================================
class GoldBollinger(BaseGoldAnalyst):
    """4H 布林带 — 黄金常沿布林带边界运行"""

    def __init__(self):
        super().__init__(name="4H-布林带(20,2)", weight=0.8)

    def analyze(self, gold_data: dict) -> dict:
        df = gold_data.get("df_4h", pd.DataFrame()).copy()
        if len(df) < 22:
            return {"action": "HOLD", "score": 0.0, "reason": "数据不足"}

        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bandwidth = (upper - lower) / sma  # 带宽(波动率)

        price = df["close"].iloc[-1]
        bb_upper = upper.iloc[-1]
        bb_lower = lower.iloc[-1]
        bb_mid = sma.iloc[-1]
        bw = bandwidth.iloc[-1]

        if pd.isna(bb_upper):
            return {"action": "HOLD", "score": 0.0, "reason": "布林带无效"}

        # 价格穿下轨 → 超卖反弹机会
        if price < bb_lower:
            return {"action": "BUY", "score": 0.60,
                    "reason": f"跌破下轨 {price:.1f}<{bb_lower:.1f}"}
        # 价格穿上轨 → 超买回落风险
        elif price > bb_upper:
            return {"action": "SELL", "score": 0.60,
                    "reason": f"突破上轨 {price:.1f}>{bb_upper:.1f}"}
        # 布林带收窄 → 即将变盘
        elif not pd.isna(bw) and bw < 0.02:
            return {"action": "HOLD", "score": 0.0,
                    "reason": f"布林带收窄(带宽{bw:.2%})，等待突破"}
        # 价格偏下轨
        elif price < bb_mid:
            return {"action": "BUY", "score": 0.15, "reason": "价格偏下轨"}
        else:
            return {"action": "SELL", "score": 0.15, "reason": "价格偏上轨"}


# ===================================================================
# 5. 4H 支撑阻力位 — 关键价位
# ===================================================================
class GoldSupportResistance(BaseGoldAnalyst):
    """4H 最近 20 根K线的高/低点作为支撑阻力"""

    def __init__(self):
        super().__init__(name="4H-支撑阻力", weight=0.7)

    def analyze(self, gold_data: dict) -> dict:
        df = gold_data.get("df_4h", pd.DataFrame()).copy()
        if len(df) < 20:
            return {"action": "HOLD", "score": 0.0, "reason": "数据不足"}

        recent = df.tail(20)
        resistance = recent["high"].max()
        support = recent["low"].min()
        price = df["close"].iloc[-1]
        pivot = (resistance + support) / 2

        dist_to_res = (resistance - price) / price * 100
        dist_to_sup = (price - support) / price * 100

        # 接近支撑位
        if dist_to_sup < 0.5:
            return {"action": "BUY", "score": 0.55,
                    "reason": f"接近支撑 {support:.1f}(距{dist_to_sup:.1f}%)"}
        # 接近阻力位
        elif dist_to_res < 0.5:
            return {"action": "SELL", "score": 0.55,
                    "reason": f"接近阻力 {resistance:.1f}(距{dist_to_res:.1f}%)"}
        # 价格在支撑区
        elif price < pivot and price > support:
            return {"action": "BUY", "score": 0.20,
                    "reason": "价格在支撑区"}
        elif price > pivot and price < resistance:
            return {"action": "SELL", "score": 0.20,
                    "reason": "价格在阻力区"}
        else:
            return {"action": "HOLD", "score": 0.0, "reason": "无明确S/R信号"}


# ===================================================================
# 6. 美元指数负相关参考
# ===================================================================
class GoldDXYCorrelation(BaseGoldAnalyst):
    """黄金与美元指数(DXY)负相关验证"""

    def __init__(self):
        super().__init__(name="美元指数关联", weight=0.6)

    def analyze(self, gold_data: dict) -> dict:
        dxy = gold_data.get("dxy", pd.DataFrame())
        gold_4h = gold_data.get("df_4h", pd.DataFrame())

        if dxy.empty or gold_4h.empty:
            return {"action": "HOLD", "score": 0.0, "reason": "DXY数据缺失"}

        try:
            # 美元指数 5 日涨跌
            dxy_5d = dxy["close"].pct_change(5).iloc[-1]
            # 黄金 4H 最后 12 根 (约2天) 涨跌
            if len(gold_4h) >= 12:
                gold_12b = (gold_4h["close"].iloc[-1] / gold_4h["close"].iloc[-12] - 1)
            else:
                gold_12b = gold_4h["close"].pct_change(len(gold_4h)-1).iloc[-1]

            if pd.isna(dxy_5d) or pd.isna(gold_12b):
                return {"action": "HOLD", "score": 0.0, "reason": "DXY数据不够"}

            # 美元跌 + 金涨 → 正常负相关，看好
            if dxy_5d < -0.01 and gold_12b > 0:
                return {"action": "BUY", "score": 0.35,
                        "reason": f"美元走弱+黄金走强 DXY{dxy_5d:.1%}"}
            # 美元涨 + 金跌 → 正常负相关，看空
            elif dxy_5d > 0.01 and gold_12b < 0:
                return {"action": "SELL", "score": 0.35,
                        "reason": f"美元走强+黄金走弱 DXY{dxy_5d:.1%}"}
            # 美元跌但金也跌 → 异常信号，警惕
            elif dxy_5d < -0.01 and gold_12b < -0.005:
                return {"action": "SELL", "score": 0.25,
                        "reason": f"异常:美元跌但黄金不涨 DXY{dxy_5d:.1%}"}
            # 美元涨但金也涨 → 强势信号
            elif dxy_5d > 0.01 and gold_12b > 0.005:
                return {"action": "BUY", "score": 0.30,
                        "reason": f"美元涨但黄金更强 Gold{gold_12b:.1%}"}
            else:
                return {"action": "HOLD", "score": 0.0, "reason": "DXY中性"}
        except Exception as e:
            return {"action": "HOLD", "score": 0.0, "reason": f"DXY异常:{e}"}


# ===================================================================
# 获取全部黄金分析师
# ===================================================================

def get_gold_analysts() -> list:
    """返回所有黄金专属分析师"""
    return [
        GoldEMA12_26(),             # 权重 1.2
        GoldRSI(),                   # 权重 1.0
        GoldMACD(),                  # 权重 1.0
        GoldBollinger(),             # 权重 0.8
        GoldSupportResistance(),     # 权重 0.7
        GoldDXYCorrelation(),        # 权重 0.6
    ]
