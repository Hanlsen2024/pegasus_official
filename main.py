"""
皮卡斯 2.0 (Picas) — AI驱动的多智能体量化信号引擎
对标 Brale 架构: 快慢双循环 + 多Agent LLM推理 + 指标计算库

用法:
  python main.py gold        # 黄金AI分析
  python main.py us -s AAPL   # 美股AI分析
  python main.py a -s 600519  # A股AI分析
"""
import json
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("picas")


def main():
    parser = argparse.ArgumentParser(
        description="皮卡斯 2.0 — AI驱动的多智能体量化信号引擎"
    )
    parser.add_argument("market", nargs="?", default="gold",
                        help="市场: gold(黄金) / us(美股) / a(A股)")
    parser.add_argument("-s", "--symbol", default=None, help="股票代码")
    parser.add_argument("--no-ai", action="store_true",
                        help="禁用AI，回退到纯数学指标模式")
    parser.add_argument("--compact", action="store_true",
                        help="紧凑输出(隐藏推理细节)")
    parser.add_argument("--serve", action="store_true",
                        help="启动API服务")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        logger.info("启动皮卡斯 API 服务 http://0.0.0.0:8000")
        uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=False)
        return

    if args.no_ai:
        # 回退到旧版纯指标分析
        from data.fetcher import get_market_data
        from data.gold_fetcher import get_gold_full
        from core.analysts import get_all_analysts as get_stock_analysts
        from core.gold_analysts import get_gold_analysts
        from core.engine import VotingEngine
        import pandas as pd, datetime as dt

        def old_run(market, symbol):
            if market == "gold":
                gold_data = get_gold_full()
                df_4h = gold_data.get("df_4h", pd.DataFrame())
                analysts = get_gold_analysts()
                signals, votes = [], []
                for a in analysts:
                    r = a.analyze(gold_data)
                    signals.append({"action": r["action"], "score": r["score"], "weight": a.weight, "reason": r.get("reason", "")})
                    votes.append({"name": a.name, "signal": r["action"], "score": round(r["score"], 4), "confidence": 0, "reason": r.get("reason", ""), "reasoning": "", "signals": []})
                engine = VotingEngine()
                vr = engine.vote(signals)
                price = float(df_4h["close"].iloc[-1]) if not df_4h.empty else 0
                return {"timestamp": dt.datetime.now().isoformat(), "market": "gold", "symbol": "GC=F", "signal": vr["action"], "net_score": round(vr.get("score", 0), 4), "buy_score": round(vr.get("buy_score", 0), 4), "sell_score": round(vr.get("sell_score", 0), 4), "price": round(price, 2), "agent_count": len(analysts), "analyst_votes": votes, "indicators_snapshot": {}}
            else:
                df = get_market_data(market=market, symbol=symbol)
                analysts = get_stock_analysts()
                signals, votes = [], []
                for a in analysts:
                    r = a.analyze({"df": df})
                    signals.append({"action": r["action"], "score": r["score"], "weight": a.weight, "reason": r.get("reason", "")})
                    votes.append({"name": a.name, "signal": r["action"], "score": round(r["score"], 4), "confidence": 0, "reason": r.get("reason", ""), "reasoning": "", "signals": []})
                engine = VotingEngine()
                vr = engine.vote(signals)
                price = float(df["close"].iloc[-1]) if "close" in df.columns else 0
                return {"timestamp": dt.datetime.now().isoformat(), "market": market, "symbol": symbol, "signal": vr["action"], "net_score": round(vr.get("score", 0), 4), "buy_score": round(vr.get("buy_score", 0), 4), "sell_score": round(vr.get("sell_score", 0), 4), "price": round(price, 2), "agent_count": len(analysts), "analyst_votes": votes, "indicators_snapshot": {}}

        result = old_run(args.market, args.symbol)
    else:
        from core.pipeline import run_ai_pipeline
        result = run_ai_pipeline(market=args.market, symbol=args.symbol)

    if args.compact:
        # 紧凑输出：只保留投票摘要
        result.pop("analyst_votes", None)
        result.pop("indicators_snapshot", None)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
