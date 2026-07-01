"""
指标计算库 — 对标 Brale 的 go-talib
支持: 趋势 / 动量 / 波动率 / 成交量 四类指标
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("picas.indicators")


# =========================================================================
# 趋势类指标
# =========================================================================

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA 指数移动平均"""
    return series.ewm(span=span, adjust=False).mean()


def calc_sma(series: pd.Series, window: int) -> pd.Series:
    """SMA 简单移动平均"""
    return series.rolling(window=window).mean()


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
              signal: int = 9) -> dict:
    """MACD 指标"""
    ema_fast = calc_ema(df["close"], fast)
    ema_slow = calc_ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


# =========================================================================
# 动量类指标
# =========================================================================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 相对强弱指标 (Wilder平滑)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_stochastic(df: pd.DataFrame, k_period: int = 14,
                    d_period: int = 3) -> dict:
    """KDJ 随机指标"""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k_raw = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k_raw.rolling(d_period).mean()  # %K
    d = k.rolling(d_period).mean()       # %D
    return {"k": k, "d": d}


def calc_roc(series: pd.Series, period: int = 10) -> pd.Series:
    """ROC 变动率 (Rate of Change)"""
    return (series / series.shift(period) - 1) * 100


# =========================================================================
# 波动率类指标
# =========================================================================

def calc_bollinger(series: pd.Series, window: int = 20,
                   num_std: float = 2.0) -> dict:
    """布林带"""
    mid = calc_sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return {"upper": upper, "lower": lower, "mid": mid, "bandwidth": bandwidth}


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR 平均真实波幅"""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# =========================================================================
# 成交量类指标
# =========================================================================

def calc_obv(df: pd.DataFrame) -> pd.Series:
    """OBV 能量潮"""
    sign = np.sign(df["close"].diff())
    obv = (sign * df["volume"]).cumsum()
    return obv


def calc_volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """成交量分布 (POC 识别)"""
    price = df["close"].values
    volume = df["volume"].values if "volume" in df.columns else np.ones_like(price)

    hist, bin_edges = np.histogram(price, bins=bins, weights=volume)
    poc_idx = np.argmax(hist)
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    return {
        "poc": poc,                    # 成交量最大价位
        "vah": bin_edges[poc_idx + 1], # 价值区域上沿
        "val": bin_edges[poc_idx],     # 价值区域下沿
    }


# =========================================================================
# 支撑阻力
# =========================================================================

def calc_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """基于最近 N 根K线计算支撑阻力"""
    recent = df.tail(lookback)
    swing_high = recent["high"].nlargest(3).tolist()
    swing_low = recent["low"].nsmallest(3).tolist()

    return {
        "resistance": round(recent["high"].max(), 2),
        "support": round(recent["low"].min(), 2),
        "pivot": round((recent["high"].max() + recent["low"].min()) / 2, 2),
        "swing_highs": [round(x, 2) for x in swing_high],
        "swing_lows": [round(x, 2) for x in swing_low],
    }


# =========================================================================
# 综合指标包 — 一次性计算全部
# =========================================================================

def compute_all_indicators(df: pd.DataFrame) -> dict:
    """
    一次性计算所有技术指标，返回结构化字典
    """
    if df.empty or len(df) < 30:
        return {"error": "数据不足，需要至少30条K线"}

    close = df["close"]

    # 趋势
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    ema50 = calc_ema(close, 50)
    sma20 = calc_sma(close, 20)
    sma50 = calc_sma(close, 50)
    macd_data = calc_macd(df)
    bb = calc_bollinger(close)

    # 动量
    rsi = calc_rsi(close, 14)
    stoch = calc_stochastic(df)
    roc10 = calc_roc(close, 10)

    # 波动率
    atr = calc_atr(df)

    # 成交量
    obv_val = None
    vol_profile = {}
    if "volume" in df.columns and df["volume"].sum() > 0:
        obv_val = calc_obv(df)
        vol_profile = calc_volume_profile(df)

    # 支撑阻力
    sr = calc_support_resistance(df)

    # 提取最新值
    def last_val(s, fmt=".2f"):
        v = s.iloc[-1] if len(s) > 0 else np.nan
        return round(float(v), 2) if not np.isnan(v) else "NaN"

    def last_diff(s):
        """最近两根的变化方向"""
        if len(s) < 2:
            return "flat"
        v, p = s.iloc[-1], s.iloc[-2]
        if pd.isna(v) or pd.isna(p):
            return "flat"
        return "up" if v > p else "down" if v < p else "flat"

    current_price = float(close.iloc[-1]) if len(close) > 0 else 0

    return {
        "price": round(current_price, 2),
        "trend": {
            "ema12": last_val(ema12), "ema26": last_val(ema26),
            "ema50": last_val(ema50),
            "sma20": last_val(sma20), "sma50": last_val(sma50),
            "ema_cross": "bullish" if last_val(ema12) > last_val(ema26) else "bearish",
            "price_vs_ema50": "above" if current_price > float(last_val(ema50)) else "below",
            "macd": {
                "macd_line": last_val(macd_data["macd"], ".4f"),
                "signal_line": last_val(macd_data["signal"], ".4f"),
                "histogram": last_val(macd_data["histogram"], ".4f"),
                "direction": last_diff(macd_data["histogram"]),
            },
        },
        "momentum": {
            "rsi14": last_val(rsi),
            "rsi_zone": "oversold" if last_val(rsi) < 30 else "overbought" if last_val(rsi) > 70 else "neutral",
            "stoch_k": last_val(stoch["k"]),
            "stoch_d": last_val(stoch["d"]),
            "roc10": last_val(roc10),
        },
        "volatility": {
            "atr14": last_val(atr),
            "atr_pct": round(float(last_val(atr)) / current_price * 100, 2) if current_price > 0 else 0,
            "bollinger": {
                "upper": last_val(bb["upper"]),
                "mid": last_val(bb["mid"]),
                "lower": last_val(bb["lower"]),
                "bandwidth_pct": f"{round(float(last_val(bb['bandwidth'])) * 100, 2)}%",
                "position": "above_upper" if current_price > float(last_val(bb["upper"]))
                else "below_lower" if current_price < float(last_val(bb["lower"]))
                else "inside",
            },
        },
        "volume": {
            "obv_trend": last_diff(obv_val) if obv_val is not None else "N/A",
            "poc": vol_profile.get("poc", "N/A"),
            "vah": vol_profile.get("vah", "N/A"),
            "val": vol_profile.get("val", "N/A"),
        } if obv_val is not None else {},
        "sr_levels": sr,
    }


# =========================================================================
# 向量化指标序列 — 回测优化用 (O(n) 替代 O(n²))
# =========================================================================

def compute_indicator_series(df: pd.DataFrame) -> dict:
    """
    一次性计算全部指标序列，返回 dict[str, pd.Series]。
    回测引擎可按索引值取值，无需每根K线重新计算全部指标。
    
    返回的每个 Series 与 df 等长，iloc[i] 即为第i根K线位置的值(无未来函数)。
    """
    close = df["close"]
    
    # 趋势
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    ema50 = calc_ema(close, 50)
    macd_data = calc_macd(df)
    bb = calc_bollinger(close)
    
    # 动量
    rsi = calc_rsi(close, 14)
    stoch = calc_stochastic(df)
    
    # 波动率
    atr = calc_atr(df)
    
    series = {
        "ema12": ema12,
        "ema26": ema26,
        "ema50": ema50,
        "macd_hist": macd_data["histogram"],
        "rsi14": rsi,
        "stoch_k": stoch["k"],
        "atr14": atr,
        "bb_upper": bb["upper"],
        "bb_mid": bb["mid"],
        "bb_lower": bb["lower"],
    }
    
    return series


def snap_indicators_at(series: dict, idx: int, price: float) -> dict:
    """
    从预计算指标序列中提取 idx 位置的快照值（对标 compute_all_indicators 输出格式）。
    """
    def val(name):
        s = series[name]
        if idx < 0 or idx >= len(s):
            return float("nan")
        return float(s.iloc[idx])
    
    def vals(keys):
        return {k: round(val(k), 2) for k in keys}
    
    ema12_val = val("ema12")
    ema26_val = val("ema26")
    ema50_val = val("ema50")
    rsi_val = val("rsi14")
    atr_val = val("atr14")
    macd_hist_val = val("macd_hist")
    
    # MACD 方向：看histogram最近两根的变化
    macd_hist_s = series["macd_hist"]
    if idx >= 1 and not pd.isna(macd_hist_s.iloc[idx]) and not pd.isna(macd_hist_s.iloc[idx - 1]):
        macd_dir = "up" if macd_hist_s.iloc[idx] > macd_hist_s.iloc[idx - 1] else "down"
    else:
        macd_dir = "flat"
    
    # 布林位置
    bb_up = val("bb_upper")
    bb_lo = val("bb_lower")
    if pd.isna(bb_up) or pd.isna(bb_lo):
        bb_pos = "inside"
    elif price > bb_up:
        bb_pos = "above_upper"
    elif price < bb_lo:
        bb_pos = "below_lower"
    else:
        bb_pos = "inside"
    
    # RSI 区域
    if pd.isna(rsi_val):
        rsi_zone = "neutral"
    elif rsi_val < 30:
        rsi_zone = "oversold"
    elif rsi_val > 70:
        rsi_zone = "overbought"
    else:
        rsi_zone = "neutral"
    
    return {
        "price": round(price, 2),
        "trend": {
            "ema12": round(ema12_val, 2),
            "ema26": round(ema26_val, 2),
            "ema50": round(ema50_val, 2),
            "ema_cross": "bullish" if ema12_val > ema26_val else "bearish",
            "price_vs_ema50": "above" if price > ema50_val else "below",
            "macd": {
                "macd_line": "N/A",
                "signal_line": "N/A",
                "histogram": round(macd_hist_val, 4),
                "direction": macd_dir,
            },
        },
        "momentum": {
            "rsi14": round(rsi_val, 2) if not pd.isna(rsi_val) else 50.0,
            "rsi_zone": rsi_zone,
            "stoch_k": round(val("stoch_k"), 2) if not pd.isna(val("stoch_k")) else 50.0,
            "stoch_d": "N/A",
            "roc10": 0.0,
        },
        "volatility": {
            "atr14": round(atr_val, 2) if not pd.isna(atr_val) else 0.0,
            "atr_pct": round(float(atr_val) / price * 100, 2) if price > 0 and not pd.isna(atr_val) else 0,
            "bollinger": {
                "upper": round(bb_up, 2),
                "mid": round(val("bb_mid"), 2),
                "lower": round(bb_lo, 2),
                "bandwidth_pct": "N/A",
                "position": bb_pos,
            },
        },
        "volume": {},
        "sr_levels": {},
    }
