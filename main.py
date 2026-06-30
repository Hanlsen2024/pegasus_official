"""
皮卡斯 (Picas) - 多市场多分析师投票系统
黄金/美股/A股 → 各自独立的智能体管线
"""

import json
import datetime
import logging
import sys
import argparse
import traceback

from data.fetcher import get_market_data
from data.gold_fetcher import get_gold_full
from core.analysts import get_all_analysts as get_stock_analysts
from core.gold_analysts import get_gold_analysts
from core.engine import VotingEngine

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("picas")


# ---------------------------------------------------------------------------
# 统一输出格式
# ---------------------------------------------------------------------------

def _format_output(market: str, symbol: str, engine_result: dict,
                   analyst_votes: list, current_price: float = None) -> dict:
    """将投票引擎结果转为统一 API 格式"""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "market": market,
        "symbol": symbol,
        "signal": engine_result["action"],
        "net_score": round(engine_result.get("score", 0), 4),
        "buy_score": round(engine_result.get("buy_score", 0), 4),
        "sell_score": round(engine_result.get("sell_score", 0), 4),
        "price": round(current_price, 2) if current_price else None,
        "analyst_votes": analyst_votes,
        "analyst_count": len(analyst_votes),
    }


# ---------------------------------------------------------------------------
# 黄金专属分析管线 (4H + 日线)
# ---------------------------------------------------------------------------

def run_gold_analysis() -> dict:
    """黄金专属分析：多周期数据 + 6位黄金专属智能体"""
    logger.info("=" * 50)
    logger.info("🥇 启动黄金专属分析管线 (4H + 日线)")
    logger.info("=" * 50)

    # 1) 获取多周期数据
    try:
        gold_data = get_gold_full()
    except Exception as e:
        return {"timestamp": datetime.datetime.now().isoformat(), "error": f"黄金数据获取失败: {e}"}

    df_4h = gold_data.get("df_4h", pd.DataFrame())
    if df_4h.empty:
        return {"timestamp": datetime.datetime.now().isoformat(), "error": "黄金4H数据为空"}

    # 2) 黄金专属分析师
    analysts = get_gold_analysts()
    logger.info(f"黄金智能体数量: {len(analysts)}")

    # 3) 各分析师独立分析
    signals = []
    analyst_votes = []

    for analyst in analysts:
        try:
            result = analyst.analyze(gold_data)
            signals.append({
                "action": result["action"],
                "score": result["score"],
                "weight": analyst.weight,
                "reason": result.get("reason", ""),
            })
            analyst_votes.append({
                "name": analyst.name,
                "signal": result["action"],
                "score": round(result["score"], 4),
                "reason": result.get("reason", ""),
            })
            logger.info(f"  {analyst.name:28s} → {result['action']:4s} "
                       f"score={result['score']:.3f}  {result.get('reason','')}")
        except Exception as e:
            logger.error(f"分析师 {analyst.name} 出错: {e}")
            signals.append({"action": "HOLD", "score": 0.0, "weight": analyst.weight})
            analyst_votes.append({"name": analyst.name, "signal": "HOLD", "score": 0, "reason": f"异常:{e}"})

    # 4) 投票
    engine = VotingEngine()
    result = engine.vote(signals)

    # 5) 价格
    price = float(df_4h["close"].iloc[-1]) if "close" in df_4h.columns else None

    return _format_output("gold", "GC=F", result, analyst_votes, price)


# ---------------------------------------------------------------------------
# 股票分析管线 (日线)
# ---------------------------------------------------------------------------

def run_stock_analysis(market: str, symbol: str) -> dict:
    """股票分析：日线数据 + 5位通用股票智能体"""
    logger.info("=" * 50)
    logger.info(f"📊 启动股票分析管线 market={market} symbol={symbol}")
    logger.info("=" * 50)

    # 1) 获取数据
    try:
        import pandas as pd
        df = get_market_data(market=market, symbol=symbol)
    except Exception as e:
        return {"timestamp": datetime.datetime.now().isoformat(), "error": f"数据获取失败: {e}"}

    if df is None or df.empty:
        return {"timestamp": datetime.datetime.now().isoformat(), "error": "获取数据为空"}

    # 2) 股票分析师
    analysts = get_stock_analysts()
    logger.info(f"股票智能体数量: {len(analysts)}")

    # 3) 分析
    signals = []
    analyst_votes = []

    for analyst in analysts:
        try:
            result = analyst.analyze({"df": df})
            signals.append({
                "action": result["action"],
                "score": result["score"],
                "weight": analyst.weight,
                "reason": result.get("reason", ""),
            })
            analyst_votes.append({
                "name": analyst.name,
                "signal": result["action"],
                "score": round(result["score"], 4),
                "reason": result.get("reason", ""),
            })
        except Exception as e:
            logger.error(f"分析师 {analyst.name} 出错: {e}")
            signals.append({"action": "HOLD", "score": 0.0, "weight": analyst.weight})
            analyst_votes.append({"name": analyst.name, "signal": "HOLD", "score": 0, "reason": f"异常:{e}"})

    # 4) 投票
    engine = VotingEngine()
    result = engine.vote(signals)

    # 5) 价格
    price = float(df["close"].iloc[-1]) if "close" in df.columns else None

    return _format_output(market, symbol, result, analyst_votes, price)


# ---------------------------------------------------------------------------
# Mock 测试管线
# ---------------------------------------------------------------------------

def run_mock_analysis() -> dict:
    """离线测试"""
    from data.fetcher import get_mock_data
    import pandas as pd

    df = get_mock_data()
    analysts = get_stock_analysts()
    signals = []
    analyst_votes = []

    for analyst in analysts:
        try:
            result = analyst.analyze({"df": df})
            signals.append({"action": result["action"], "score": result["score"],
                           "weight": analyst.weight, "reason": result.get("reason", "")})
            analyst_votes.append({"name": analyst.name, "signal": result["action"],
                                  "score": round(result["score"], 4), "reason": result.get("reason", "")})
        except Exception as e:
            signals.append({"action": "HOLD", "score": 0.0, "weight": analyst.weight})
            analyst_votes.append({"name": analyst.name, "signal": "HOLD", "score": 0, "reason": f"异常:{e}"})

    engine = VotingEngine()
    result = engine.vote(signals)
    price = float(df["close"].iloc[-1])
    return _format_output("mock", None, result, analyst_votes, price)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def run_analysis(market: str = "gold", symbol: str = None) -> dict:
    """
    统一分析入口，按市场分流到不同智能体管线

    Args:
        market: "gold" / "us" / "a" / "mock"
        symbol: 股票代码
    """
    try:
        if market == "gold":
            return run_gold_analysis()
        elif market == "mock":
            return run_mock_analysis()
        else:
            return run_stock_analysis(market, symbol)
    except Exception as e:
        logger.exception("分析异常")
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": f"分析失败: {e}",
        }


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="皮卡斯 - 多市场交易信号分析")
    parser.add_argument("market", nargs="?", default="gold",
                        choices=["gold", "us", "a", "mock"],
                        help="gold(黄金)/us(美股)/a(A股)/mock(测试)")
    parser.add_argument("-s", "--symbol", default=None, help="股票代码")
    parser.add_argument("--compact", action="store_true")

    args = parser.parse_args()
    symbol = args.symbol

    # 自动推断
    if not symbol and args.market not in ("gold", "us", "a", "mock"):
        symbol = args.market
        args.market = "a" if (symbol.isdigit() and len(symbol) == 6) else "us"

    output = run_analysis(market=args.market, symbol=symbol)
    if args.compact:
        output.pop("analyst_votes", None)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
