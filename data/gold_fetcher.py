"""
黄金数据获取 — MT5 优先 (实时+4H原生支持) → akshare 备用
"""
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_gold_4h() -> pd.DataFrame:
    """获取黄金 4H K线 (MT5优先)"""
    # ---- 尝试 MT5 ----
    try:
        from data.mt5_fetcher import get_mt5_rates, init_mt5
        if init_mt5():
            df = get_mt5_rates("XAUUSD", "4h", 200)
            if not df.empty:
                logger.info(f"MT5 黄金 4H: {len(df)} 条")
                return df
    except Exception as e:
        logger.warning(f"MT5 不可用: {e}")

    # ---- 备用 akshare ----
    logger.info("MT5 不可用，使用 akshare 日线数据")
    return _get_ak_gold()


def get_gold_daily() -> pd.DataFrame:
    """获取黄金日线"""
    try:
        from data.mt5_fetcher import get_mt5_rates, init_mt5
        if init_mt5():
            df = get_mt5_rates("XAUUSD", "1d", 120)
            if not df.empty:
                return df
    except Exception:
        pass
    return _get_ak_gold()


def get_dxy_daily() -> pd.DataFrame:
    """美元指数"""
    try:
        from data.mt5_fetcher import get_mt5_rates, init_mt5
        if init_mt5():
            df = get_mt5_rates("US30", "1d", 120)
            if not df.empty:
                return df
    except Exception:
        pass
    return pd.DataFrame()


def get_gold_full() -> dict:
    """黄金多周期数据包"""
    return {
        "df_4h": get_gold_4h(),
        "df_daily": get_gold_daily(),
        "dxy": get_dxy_daily(),
    }


def _get_ak_gold() -> pd.DataFrame:
    """akshare 上期所黄金期货日线 (备用)"""
    try:
        import akshare as ak
        df = ak.futures_zh_daily_sina(symbol="AU0")
        if df is None or df.empty:
            return pd.DataFrame()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = df.get("close", 0) if "close" in df.columns else 0
        return df.sort_index()
    except Exception as e:
        logger.warning(f"akshare 黄金失败: {e}")
        return pd.DataFrame()
