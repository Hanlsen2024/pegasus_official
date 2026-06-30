"""
皮卡斯 (Picas) - 多市场多分析师投票系统
支持: 黄金期货 / 美股 / A股
"""

import json
import datetime
import logging
import sys
import argparse
import traceback

from data.fetcher import get_market_data
from core.analysts import get_all_analysts
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
# 核心分析流程
# ---------------------------------------------------------------------------

def run_analysis(market: str = "gold", symbol: str = None) -> dict:
    """
    执行一轮完整分析

    Args:
        market: "gold" / "us" / "a"
        symbol: 股票代码 (美股如 "AAPL", A股如 "600519" 或 "贵州茅台")

    Returns:
        {"timestamp": str, "action": str, "score": float,
         "price": float, "details": list}
    """
    # 1) 获取数据
    try:
        df = get_market_data(market=market, symbol=symbol)
    except Exception as e:
        logger.error(f"数据获取异常: {e}")
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": f"数据获取失败: {e}",
        }

    if df.empty:
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": "获取数据为空，请检查网络或标的代码",
        }

    # 2) 初始化分析师
    try:
        analysts = get_all_analysts()
    except Exception as e:
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": f"分析师初始化失败: {e}",
        }

    # 3) 各分析师独立分析
    signals = []
    market_data = {"df": df}

    for analyst in analysts:
        try:
            result = analyst.analyze(market_data)
            result["analyst"] = analyst.name
            result["weight"] = analyst.weight
            signals.append(result)
            logger.info(
                f"  {analyst.name:12s} → {result['action']:4s} "
                f"score={result['score']:.3f}  {result.get('reason', '')}"
            )
        except Exception as e:
            logger.error(f"分析师 {analyst.name} 出错: {e}")
            traceback.print_exc()
            signals.append({
                "action": "HOLD", "score": 0.0, "weight": analyst.weight,
                "analyst": analyst.name, "reason": f"异常: {e}",
            })

    # 4) 投票引擎
    try:
        engine = VotingEngine()
        result = engine.vote(signals)
    except Exception as e:
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": f"投票计算失败: {e}",
        }

    # 5) 汇总输出
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "market": market,
        "symbol": symbol,
        "action": result["action"],
        "score": result["score"],
        "buy_score": result.get("buy_score", 0),
        "sell_score": result.get("sell_score", 0),
        "price": float(df["close"].iloc[-1]) if "close" in df.columns else None,
        "details": result.get("details", []),
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="皮卡斯 - 多市场交易信号分析")
    parser.add_argument("market", nargs="?", default="gold",
                        choices=["gold", "us", "a", "us_stock", "a_stock", "mock"],
                        help="市场类型: gold(黄金) / us(美股) / a(A股) / mock(离线测试)")
    parser.add_argument("-s", "--symbol", default=None,
                        help="股票代码 (美股: AAPL/TSLA, A股: 600519/贵州茅台)")
    parser.add_argument("--compact", action="store_true",
                        help="精简输出(不含details)")

    args = parser.parse_args()
    symbol = args.symbol

    # 若 market 看起来像股票代码，自动推断
    if not symbol and args.market not in ("gold", "us", "a", "us_stock", "a_stock", "mock"):
        symbol = args.market
        # 按前缀推断市场
        if symbol.isdigit() and len(symbol) == 6:
            args.market = "a"
        else:
            args.market = "us"

    try:
        output = run_analysis(market=args.market, symbol=symbol)
    except Exception as e:
        output = {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": f"未知异常: {e}",
        }
        logger.exception("主流程异常")

    if args.compact:
        del output["details"]

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
