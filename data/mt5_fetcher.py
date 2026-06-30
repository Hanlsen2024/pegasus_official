"""
MT5 数据源 — 直连 MetaTrader 5 获取实时/历史 K 线
支持: 黄金(XAUUSD) / 美股 / 外汇 / 指数
"""
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# MT5 品种映射
SYMBOL_MAP = {
    "gold":  "XAUUSD",       # 黄金现货
    "xauusd":"XAUUSD",
    "eurusd":"EURUSD",
    "gbpusd":"GBPUSD",
    "usdjpy":"USDJPY",
    "spx":   "US500",        # 标普500 (视经纪商而定)
    "nas":   "US100",        # 纳斯达克
    "dji":   "US30",         # 道琼斯
    "oil":   "USOIL",        # 原油
}
# 美股: 直接用小写代码，MT5 通常支持 .前缀 如 #AAPL
US_STOCK_MT5 = ["AAPL","TSLA","MSFT","NVDA","AMZN","GOOGL","META","SPY","QQQ"]

_mt5 = None

def _get_mt5():
    global _mt5
    if _mt5 is None:
        try:
            import MetaTrader5 as mt5
            _mt5 = mt5
        except ImportError:
            raise ImportError("请安装 MT5 Python 库: pip install MetaTrader5")
    return _mt5

def init_mt5() -> bool:
    """初始化 MT5 连接 (需 MT5 客户端已启动并登录)"""
    mt5 = _get_mt5()
    if not mt5.initialize():
        logger.error(f"MT5 初始化失败: {mt5.last_error()}")
        return False
    logger.info("MT5 连接成功")
    return True

def get_mt5_rates(symbol: str, timeframe: str = "1h", bars: int = 200) -> pd.DataFrame:
    """
    从 MT5 获取 K 线数据
    
    Args:
        symbol: MT5 品种代码 (XAUUSD / AAPL / etc)
        timeframe: "1m" "5m" "15m" "1h" "4h" "1d" "1w"
        bars: 获取多少根K线
        
    Returns:
        pd.DataFrame (columns: open, high, low, close, volume)
    """
    mt5 = _get_mt5()
    
    if not init_mt5():
        return pd.DataFrame()
    
    # 时间框架映射
    tf_map = {
        "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1, "1w": mt5.TIMEFRAME_W1,
    }
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    # 解析美股代码
    mt5_symbol = _resolve_symbol(symbol)
    
    logger.info(f"MT5 获取数据: {mt5_symbol} {timeframe} x{bars}")
    rates = mt5.copy_rates_from_pos(mt5_symbol, tf, 0, bars)
    
    if rates is None or len(rates) == 0:
        logger.warning(f"MT5 返回空数据: {mt5_symbol} | {mt5.last_error()}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={
        "time": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "tick_volume": "volume",
    })
    df = df.set_index("date")
    
    # 确保必要列
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = df.get("close", 0) if "close" in df.columns else 0
    
    logger.info(f"MT5 数据获取成功: {mt5_symbol} {len(df)} 条")
    return df


def _resolve_symbol(symbol: str) -> str:
    """解析为 MT5 品种代码"""
    upper = symbol.upper()
    # 黄金/外汇
    if upper in SYMBOL_MAP:
        return SYMBOL_MAP[upper]
    # 美股
    if upper in US_STOCK_MT5:
        return upper
    return upper


def get_gold_mt5(timeframe: str = "4h", bars: int = 200) -> pd.DataFrame:
    """MT5 获取黄金数据"""
    return get_mt5_rates("XAUUSD", timeframe, bars)


def get_gold_full_mt5() -> dict:
    """MT5 黄金多周期数据"""
    return {
        "df_4h": get_mt5_rates("XAUUSD", "4h", 200),
        "df_daily": get_mt5_rates("XAUUSD", "1d", 120),
        "dxy": get_mt5_rates("US30", "1d", 120) if _is_available("US30") else pd.DataFrame(),
    }


def get_stock_mt5(symbol: str, timeframe: str = "1d", bars: int = 120) -> pd.DataFrame:
    """MT5 获取股票数据"""
    return get_mt5_rates(symbol, timeframe, bars)


def _is_available(symbol: str) -> bool:
    """检查品种是否可用"""
    mt5 = _get_mt5()
    if not init_mt5():
        return False
    info = mt5.symbol_info(symbol)
    return info is not None


def is_mt5_available() -> bool:
    """检查 MT5 是否可用"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            mt5.shutdown()
            return False
        mt5.shutdown()
        return True
    except ImportError:
        return False
    except Exception:
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"MT5 可用: {is_mt5_available()}")
    if is_mt5_available():
        df = get_gold_mt5("4h", 10)
        print(df)
