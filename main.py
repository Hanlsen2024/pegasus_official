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
    parser.add_argument("--compact", action="store_true",
                        help="紧凑输出(隐藏推理细节)")
    parser.add_argument("--serve", action="store_true",
                        help="启动API服务")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"启动皮卡斯 API 服务 http://0.0.0.0:{port}")
        uvicorn.run("api.index:app", host="0.0.0.0", port=port, reload=False)
        return

    from core.pipeline import run_ai_pipeline
    result = run_ai_pipeline(market=args.market, symbol=args.symbol)

    if args.compact:
        result.pop("analyst_votes", None)
        result.pop("indicators_snapshot", None)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
