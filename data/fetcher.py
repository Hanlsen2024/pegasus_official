"""
通用数据获取模块
支持: 黄金期货 / 美股 / A股
"""

import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"


# ===================================================================
# 底层：yfinance 通用获取
# ===================================================================

def _fetch_yfinance(symbol: str, period: str = DEFAULT_PERIOD,
                    interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """yfinance 通用抓取"""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance 未安装，请执行: pip install yfinance")
        return pd.DataFrame()

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"yfinance 返回空数据，symbol={symbol}")
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        logger.info(f"yfinance 获取成功: {symbol}, {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"yfinance 获取失败 [{symbol}]: {e}")
        return pd.DataFrame()


# ===================================================================
# 底层：akshare 通用获取
# ===================================================================

def _fetch_akshare_stock(symbol: str, market: str = "sh") -> pd.DataFrame:
    """
    akshare A股个股历史数据

    Args:
        symbol: 股票代码，如 "000001"、"600519"
        market: "sh"(上海) 或 "sz"(深圳)
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare 未安装，请执行: pip install akshare")
        return pd.DataFrame()

    try:
        # 拼接 akshare 格式: sh600519 / sz000001
        full_code = f"{market}{symbol}"

        # A股个股日线
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=(datetime.now() - timedelta(days=200)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
            adjust="qfq",  # 前复权
        )

        if df.empty:
            logger.warning(f"akshare 返回空数据，symbol={full_code}")
            return pd.DataFrame()

        # 标准化列名
        col_map = {
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
            "涨跌额": "change", "换手率": "turnover",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        logger.info(f"akshare 获取成功: {full_code}, {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"akshare 获取失败 [{symbol}]: {e}")
        return pd.DataFrame()


# ===================================================================
# 1. 黄金数据
# ===================================================================

GOLD_SYMBOLS = {
    "yfinance": "GC=F",       # COMEX 黄金期货
    "akshare": "au2506",      # 上期所黄金期货
}


def get_gold_data(source: str = "auto") -> pd.DataFrame:
    """
    获取黄金数据，自动选择可用数据源

    Args:
        source: "auto" / "yfinance" / "akshare"

    Returns:
        pd.DataFrame (columns: open, high, low, close, volume)
    """
    logger.info(f"获取黄金数据，source={source}")

    if source == "yfinance":
        return _fetch_yfinance(GOLD_SYMBOLS["yfinance"])
    if source == "akshare":
        df = _fetch_akshare_stock(GOLD_SYMBOLS["akshare"])
        return df

    # auto: 优先 yfinance，失败试 akshare
    df = _fetch_yfinance(GOLD_SYMBOLS["yfinance"])
    if df.empty:
        logger.info("yfinance 失败，尝试 akshare ...")
        df = _fetch_akshare_stock(GOLD_SYMBOLS["akshare"])
    if df.empty:
        logger.error("所有数据源均无法获取黄金数据")
    return df


# ===================================================================
# 2. 美股数据
# ===================================================================

# 常见美股标的映射 {简称: yfinance代码}
US_STOCK_MAP = {
    "aapl":  "AAPL",
    "goog":  "GOOGL",
    "msft":  "MSFT",
    "amzn":  "AMZN",
    "tsla":  "TSLA",
    "nvda":  "NVDA",
    "meta":  "META",
    "spy":   "SPY",       # 标普500 ETF
    "qqq":   "QQQ",       # 纳斯达克100 ETF
    "dji":   "DIA",       # 道琼斯 ETF
    "gld":   "GLD",       # 黄金 ETF
    "uso":   "USO",       # 原油 ETF
}


def get_us_stock_data(symbol: str) -> pd.DataFrame:
    """
    获取美股个股/ETF 数据 (优先 akshare，备用 yfinance)

    Args:
        symbol: 美股代码 或 简称 (如 "AAPL" / "aapl")

    Returns:
        pd.DataFrame (columns: open, high, low, close, volume)
    """
    code = US_STOCK_MAP.get(symbol.lower(), symbol.upper())
    logger.info(f"获取美股数据: {code}")

    # 优先用 akshare (国内直连)
    try:
        import akshare as ak
        df = ak.stock_us_hist(symbol=code, period="daily",
                              start_date=(datetime.now() - timedelta(days=200)).strftime("%Y%m%d"),
                              end_date=datetime.now().strftime("%Y%m%d"),
                              adjust="qfq")
        if df is not None and not df.empty:
            # 标准化列名
            col_map = {
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            logger.info(f"akshare 美股获取成功: {code}, {len(df)} 条")
            return df
    except Exception as e:
        logger.warning(f"akshare 美股失败: {e}")

    # 备用 yfinance
    return _fetch_yfinance(code)


# ===================================================================
# 3. A股数据
# ===================================================================

# 常见A股标的映射 {简称: (代码, 市场)}
A_STOCK_MAP = {
    "贵州茅台":  ("600519", "sh"),
    "宁德时代":  ("300750", "sz"),
    "比亚迪":    ("002594", "sz"),
    "招商银行":  ("600036", "sh"),
    "中国平安":  ("601318", "sh"),
    "上证50":    ("510050", "sh"),   # 上证50 ETF
    "沪深300":   ("510300", "sh"),   # 沪深300 ETF
    "创业板":    ("159915", "sz"),   # 创业板 ETF
    "科创50":    ("588000", "sh"),   # 科创50 ETF
    "恒生互联":  ("513330", "sh"),   # 恒生互联网 ETF
}


def get_a_stock_data(symbol: str) -> pd.DataFrame:
    """
    获取A股数据

    Args:
        symbol: 股票代码(如 "600519") 或 名称(如 "贵州茅台")

    Returns:
        pd.DataFrame (columns: open, high, low, close, volume)
    """
    if symbol in A_STOCK_MAP:
        code, market = A_STOCK_MAP[symbol]
    else:
        # 根据代码前缀自动判断市场
        code = symbol
        if code.startswith(("6", "5")):
            market = "sh"
        else:
            market = "sz"

    logger.info(f"获取A股数据: {market}{code}")
    return _fetch_akshare_stock(code, market)


# ===================================================================
# 4. 数字货币 (yfinance 支持 BTC-USD / ETH-USD 等)
# ===================================================================

# 常见加密货币映射 {简称: yfinance代码}
CRYPTO_MAP = {
    "btc":    "BTC-USD",
    "eth":    "ETH-USD",
    "sol":    "SOL-USD",
    "bnb":    "BNB-USD",
    "doge":   "DOGE-USD",
    "xrp":    "XRP-USD",
    "avax":   "AVAX-USD",
    "ada":    "ADA-USD",
    "dot":    "DOT-USD",
    "matic":  "MATIC-USD",
    "link":   "LINK-USD",
    "uni":    "UNI-USD",
    "atom":   "ATOM-USD",
}

# yfinance 支持的加密周期
CRYPTO_INTERVALS = {
    "1h": "1h",
    "4h": "1h",   # yfinance 不支持4h，用1h替代
    "1d": "1d",
    "1w": "1wk",
    "1mo": "1mo",
}

CRYPTO_PERIOD_MAP = {
    "1h": "7d",    # 1小时线取7天数据
    "4h": "30d",   # 4小时用1h数据取30天
    "1d": "6mo",
    "1w": "1y",
    "1mo": "2y",
}


def get_crypto_data(symbol: str, interval: str = "4h") -> pd.DataFrame:
    """
    获取数字货币历史数据

    Args:
        symbol: 如 "BTC-USD" / "btc" / "BTC"
        interval: "1h" / "4h" / "1d"

    Returns:
        pd.DataFrame (columns: open, high, low, close, volume)
    """
    # 标准化代码
    code = CRYPTO_MAP.get(symbol.lower(), symbol.upper())
    if "-" not in code and "/" not in code:
        code = f"{code}-USD"

    yf_interval = CRYPTO_INTERVALS.get(interval, "1d")
    period = CRYPTO_PERIOD_MAP.get(interval, "6mo")

    logger.info(f"获取数字货币数据: {code} interval={yf_interval} period={period}")

    try:
        import yfinance as yf
        ticker = yf.Ticker(code)
        df = ticker.history(period=period, interval=yf_interval)

        if df.empty:
            logger.warning(f"数字货币数据为空: {code}")
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        logger.info(f"数字货币获取成功: {code}, {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"数字货币数据获取失败 [{code}]: {e}")
        return pd.DataFrame()


# ===================================================================
# 5. 离线模拟数据 (测试用)
# ===================================================================

def get_mock_data(length: int = 120) -> pd.DataFrame:
    """
    生成模拟 K 线数据，用于离线测试分析逻辑
    """
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=length, freq="B")
    close = 100.0

    prices = [close]
    for _ in range(length - 1):
        change = np.random.randn() * 1.5
        close = close + change
        prices.append(close)

    df = pd.DataFrame({
        "open":  [p + np.random.randn() * 0.3 for p in prices],
        "high":  [p + abs(np.random.randn()) * 1.0 for p in prices],
        "low":   [p - abs(np.random.randn()) * 1.0 for p in prices],
        "close": prices,
        "volume": [abs(np.random.randn()) * 1000000 + 500000 for _ in prices],
    }, index=dates)

    logger.info(f"生成模拟数据: {length} 条")
    return df


# ===================================================================
# 6. 统一入口
# ===================================================================

MARKET_REGISTRY = {
    "gold":      get_gold_data,
    "us":        get_us_stock_data,
    "a":         get_a_stock_data,
    "us_stock":  get_us_stock_data,
    "a_stock":   get_a_stock_data,
    "crypto":    get_crypto_data,
    "mock":      get_mock_data,
}


def get_market_data(market: str = "gold", symbol: str = None,
                    interval: str = "1d") -> pd.DataFrame:
    """
    统一数据获取入口

    Args:
        market: "gold" / "us" / "a" / "crypto"
        symbol: 具体股票/币种代码 (gold 时忽略)
        interval: 时间周期 ("1h" / "4h" / "1d") — crypto 时有效

    Returns:
        pd.DataFrame (columns: open, high, low, close, volume)
    """
    logger.info(f"获取市场数据: market={market}, symbol={symbol}")
    fetcher = MARKET_REGISTRY.get(market)
    if fetcher is None:
        logger.error(f"不支持的市场类型: {market}，可选: {list(MARKET_REGISTRY.keys())}")
        return pd.DataFrame()

    if market in ("gold", "mock"):
        return fetcher()
    elif market == "crypto":
        if not symbol:
            logger.error("crypto 需要指定 symbol")
            return pd.DataFrame()
        return fetcher(symbol, interval)
    else:
        if not symbol:
            logger.error(f"market={market} 需要指定 symbol")
            return pd.DataFrame()
        return fetcher(symbol)


# ===================================================================
# 测试
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n===== 黄金 =====")
    df = get_gold_data()
    print(df.tail(3) if not df.empty else "无数据")

    print("\n===== 美股 AAPL =====")
    df = get_us_stock_data("AAPL")
    print(df.tail(3) if not df.empty else "无数据")

    print("\n===== A股 贵州茅台 =====")
    df = get_a_stock_data("贵州茅台")
    print(df.tail(3) if not df.empty else "无数据")
