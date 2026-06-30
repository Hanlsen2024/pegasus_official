"""
黄金专属数据获取模块 - 支持 4H / 1D 多周期
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# COMEX 黄金期货 (yfinance)
GOLD_SYMBOL = "GC=F"
# 美元指数 (用于参考)
DXY_SYMBOL = "DX-Y.NYB"


def _get_yf_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """yfinance 通用抓取"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"yfinance [{symbol}] 失败: {e}")
        return pd.DataFrame()


def get_gold_4h() -> pd.DataFrame:
    """
    获取黄金 4 小时 K 线数据
    原理：yfinance 拉 1H 数据 → resample 为 4H
    
    Returns:
        pd.DataFrame (columns: open, high, low, close, volume)
    """
    logger.info("获取黄金 4H 数据...")

    # 拉近 30 天 1H 数据，足够生成 4H K线
    df_1h = _get_yf_data(GOLD_SYMBOL, period="30d", interval="1h")

    if df_1h.empty:
        logger.warning("yfinance 1H 数据为空，降级为日线")
        return _get_yf_data(GOLD_SYMBOL, period="6mo", interval="1d")

    # Resample: 1H → 4H
    df_4h = df_1h.resample("4h").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume": "sum",
    })
    df_4h = df_4h.dropna()
    logger.info(f"黄金 4H 数据: {len(df_4h)} 条K线")
    return df_4h


def get_gold_daily() -> pd.DataFrame:
    """获取黄金日线数据（补充视角）"""
    return _get_yf_data(GOLD_SYMBOL, period="6mo", interval="1d")


def get_dxy_daily() -> pd.DataFrame:
    """获取美元指数日线（用于黄金负相关验证）"""
    return _get_yf_data(DXY_SYMBOL, period="6mo", interval="1d")


def get_gold_full() -> dict:
    """
    获取黄金多周期数据包
    
    Returns:
        {
            "df_4h": pd.DataFrame,      # 4小时K线（主力分析周期）
            "df_daily": pd.DataFrame,   # 日线（辅助确认）
            "dxy": pd.DataFrame,        # 美元指数（宏观参考）
        }
    """
    return {
        "df_4h": get_gold_4h(),
        "df_daily": get_gold_daily(),
        "dxy": get_dxy_daily(),
    }


# =====================================================================
# 测试
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_gold_full()
    for k, v in data.items():
        print(f"\n{k}: {len(v)} 条")
        if not v.empty:
            print(v.tail(2))
