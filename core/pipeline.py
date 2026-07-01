"""
皮卡斯 2.0 核心分析管线 — 对标 Brale Dual-Loop
Slow Loop: 数据获取 → 指标计算 → AI推理 → 投票 → 信号输出
"""
import logging
import concurrent.futures
from datetime import datetime
from typing import Optional

import pandas as pd

from data.fetcher import get_market_data
from data.gold_fetcher import get_gold_full
from data.indicator_bank import compute_all_indicators
from core.agents.factory import create_agents
from core.engine import VotingEngine

logger = logging.getLogger("picas.pipeline")


# =========================================================================
# 多时间框架数据准备
# =========================================================================

def _prepare_gold_data() -> dict:
    """黄金多时间框架数据"""
    gold_data = get_gold_full()
    df_4h = gold_data.get("df_4h", pd.DataFrame())
    df_daily = gold_data.get("df_daily", pd.DataFrame())
    dxy = gold_data.get("dxy", pd.DataFrame())

    if df_4h.empty:
        raise ValueError("黄金4H数据为空")

    # 主周期 4H 指标
    indicators_4h = compute_all_indicators(df_4h)
    # 辅助周期日线指标
    indicators_daily = compute_all_indicators(df_daily) if not df_daily.empty else {}

    # 价格走势摘要
    price_summary = _build_price_summary(df_4h)

    return {
        "market": "gold",
        "symbol": "GC=F",
        "price": indicators_4h.get("price", 0),
        "indicators": indicators_4h,
        "multi_timeframe": {
            "4H": _tf_summary(indicators_4h),
            "日线": _tf_summary(indicators_daily) if indicators_daily else {},
        },
        "price_summary": price_summary,
        "dxy": {
            "available": not dxy.empty,
            "last_price": round(float(dxy["close"].iloc[-1]), 2) if not dxy.empty else None,
        },
    }


def _prepare_stock_data(market: str, symbol: str) -> dict:
    """股票数据准备"""
    df = get_market_data(market=market, symbol=symbol)
    if df is None or df.empty:
        raise ValueError(f"无法获取 {market}/{symbol} 数据")

    indicators = compute_all_indicators(df)
    price_summary = _build_price_summary(df)

    return {
        "market": market,
        "symbol": symbol,
        "price": indicators.get("price", 0),
        "indicators": indicators,
        "multi_timeframe": {
            "日线": _tf_summary(indicators),
        },
        "price_summary": price_summary,
    }


def _tf_summary(indicators: dict) -> dict:
    """从指标字典提取时间框架摘要"""
    if not indicators:
        return {}
    trend = indicators.get("trend", {})
    momentum = indicators.get("momentum", {})
    return {
        "ema_cross": trend.get("ema_cross", "N/A"),
        "macd_direction": trend.get("macd", {}).get("direction", "N/A"),
        "rsi_zone": momentum.get("rsi_zone", "N/A"),
        "structure": _describe_structure(indicators),
    }


def _describe_structure(indicators: dict) -> str:
    """描述当前价格结构"""
    trend = indicators.get("trend", {})
    bb = indicators.get("volatility", {}).get("bollinger", {})
    ema_cross = trend.get("ema_cross", "")
    price_vs_ema = trend.get("price_vs_ema50", "")
    bb_pos = bb.get("position", "")

    if ema_cross == "bullish" and price_vs_ema == "above":
        return "上升趋势(Higher结构)"
    elif ema_cross == "bearish" and price_vs_ema == "below":
        return "下降趋势(Lower结构)"
    elif bb_pos in ("above_upper", "below_lower"):
        return "极端位置(潜在反转)"
    else:
        return "震荡/整理"


def _build_price_summary(df: pd.DataFrame, lookback: int = 10) -> str:
    """构建近期价格走势文本摘要"""
    if len(df) < lookback:
        return "数据不足"

    recent = df.tail(lookback)
    opens = recent["open"].tolist()
    closes = recent["close"].tolist()
    highs = recent["high"].tolist()
    lows = recent["low"].tolist()

    candles = []
    for i in range(len(closes)):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        if c >= o:
            candle_type = "阳线📈" if (c - o) / o * 100 > 1 else "小阳线"
        else:
            candle_type = "阴线📉" if (o - c) / o * 100 > 1 else "小阴线"
        candles.append(f"K{i+1}: {candle_type} O={o:.2f} C={c:.2f} H={h:.2f} L={l:.2f}")

    # 计算关键变化
    first_close = closes[0]
    last_close = closes[-1]
    change_pct = (last_close - first_close) / first_close * 100
    direction = "上涨" if change_pct > 0 else "下跌"

    summary = f"近{lookback}根K线: 整体{direction}{abs(change_pct):.2f}%。\n"
    summary += "\n".join(candles)
    return summary


# =========================================================================
# AI 推理管线
# =========================================================================

def run_ai_pipeline(market: str = "gold", symbol: str = None) -> dict:
    """
    AI 推理管线 — 对标 Brale Slow Loop

    1. 拉取多周期数据
    2. 计算全套技术指标
    3. 调度 Agent 并发推理
    4. 加权投票
    5. 输出结构化信号 + AI推理过程
    """
    logger.info("=" * 60)
    logger.info(f"🧠 皮卡斯 AI推理管线启动 | market={market} symbol={symbol}")
    logger.info("=" * 60)

    # ---- Step 1: 准备数据 ----
    try:
        if market == "gold":
            data_pack = _prepare_gold_data()
        elif market in ("us", "a"):
            if not symbol:
                return {"error": f"market={market} 需要指定 symbol", "timestamp": datetime.now().isoformat()}
            data_pack = _prepare_stock_data(market, symbol)
        else:
            return {"error": f"不支持的市场: {market}", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        return {"error": f"数据获取失败: {e}", "timestamp": datetime.now().isoformat()}

    # ---- Step 2: 创建 Agent ----
    agents = create_agents(market)
    logger.info(f"激活 {len(agents)} 个AI Agent: {[a.name for a in agents]}")

    # ---- Step 3: Agent 并发推理 ----
    agent_results = []
    analyst_votes = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(agents), 5)) as executor:
        future_map = {
            executor.submit(agent.analyze, data_pack, market): agent
            for agent in agents
        }
        for future in concurrent.futures.as_completed(future_map):
            agent = future_map[future]
            try:
                result = future.result(timeout=60)
                agent_results.append({
                    "agent": agent.name,
                    "weight": agent.weight,
                    "action": result["action"],
                    "score": result["score"],
                    "confidence": result.get("confidence", 0),
                    "reason": result.get("reason", ""),
                    "reasoning": result.get("reasoning", ""),
                    "signals": result.get("signals", []),
                    "key_level": result.get("key_level", ""),
                })
                analyst_votes.append({
                    "name": agent.name,
                    "signal": result["action"],
                    "score": result["score"],
                    "confidence": result.get("confidence", 0),
                    "reason": result.get("reason", ""),
                    "reasoning": result.get("reasoning", ""),
                    "signals": result.get("signals", []),
                })
            except Exception as e:
                logger.error(f"Agent [{agent.name}] 执行失败: {e}")
                analyst_votes.append({
                    "name": agent.name,
                    "signal": "HOLD",
                    "score": 0,
                    "confidence": 0,
                    "reason": f"执行失败: {e}",
                    "reasoning": "",
                    "signals": [],
                })

    # ---- Step 4: 加权投票 ----
    engine = VotingEngine()
    signals_for_vote = [
        {
            "action": r["action"],
            "score": abs(r["score"]),
            "weight": r["weight"],
            "reason": r["reason"],
        }
        for r in agent_results
    ]
    vote_result = engine.vote(signals_for_vote)

    # ---- Step 5: 结构化输出 ----
    price = data_pack.get("price", 0)

    output = {
        "timestamp": datetime.now().isoformat(),
        "market": market,
        "symbol": data_pack.get("symbol", symbol),
        "signal": vote_result["action"],
        "net_score": round(vote_result.get("score", 0), 4),
        "buy_score": round(vote_result.get("buy_score", 0), 4),
        "sell_score": round(vote_result.get("sell_score", 0), 4),
        "price": round(price, 2) if price else None,
        "agent_count": len(agents),
        "analyst_votes": analyst_votes,
        "indicators_snapshot": {
            "rsi14": data_pack.get("indicators", {}).get("momentum", {}).get("rsi14", "N/A"),
            "ema_cross": data_pack.get("indicators", {}).get("trend", {}).get("ema_cross", "N/A"),
            "macd_direction": data_pack.get("indicators", {}).get("trend", {}).get("macd", {}).get("direction", "N/A"),
            "bb_position": data_pack.get("indicators", {}).get("volatility", {}).get("bollinger", {}).get("position", "N/A"),
        },
    }

    logger.info(f"🎯 最终信号: {output['signal']} net={output['net_score']}")
    return output


# =========================================================================
# 交易管线 — 操盘手 + 风控师 (完整闭环)
# =========================================================================

def run_trade_pipeline(market: str = "gold", symbol: str = None,
                       portfolio: dict = None) -> dict:
    """
    完整交易管线：3个分析师 → 操盘手 → 风控师
    
    Args:
        market: 市场类型
        symbol: 股票代码
        portfolio: 账户信息 {"equity": 10000, "risk_per_trade": 0.02, 
                            "position": None|{...}}
    
    Returns:
        {"analysis": {...}, "trader": {...}, "risk_manager": {...}, "summary": str}
    """
    # ---- Step 1: 运行 AI 分析管线 ----
    analysis = run_ai_pipeline(market=market, symbol=symbol)
    
    if "error" in analysis:
        return {"error": analysis["error"], "timestamp": analysis["timestamp"]}
    
    portfolio = portfolio or {"equity": 10000, "risk_per_trade": 0.02, "position": None}
    
    # ---- Step 2: 准备操盘手数据 ----
    if market == "gold":
        data_pack = _prepare_gold_data()
    elif market in ("us", "a"):
        data_pack = _prepare_stock_data(market, symbol)
    else:
        return {"error": f"不支持的市场: {market}"}
    
    # 注入分析师结果和端口folio
    data_pack["analyst_votes"] = analysis.get("analyst_votes", [])
    data_pack["net_score"] = analysis.get("net_score", 0)
    data_pack["signal"] = analysis.get("signal", "HOLD")
    data_pack["portfolio"] = portfolio
    
    # 提取 ATR 和 RSI
    indicators = data_pack.get("indicators", {})
    data_pack["atr"] = indicators.get("volatility", {}).get("atr14", 0)
    data_pack["atr_pct"] = indicators.get("volatility", {}).get("atr_pct", 0)
    data_pack["rsi"] = indicators.get("momentum", {}).get("rsi14", 50)
    
    # ---- Step 3: 操盘手决策 ----
    from core.agents.trader_agent import TraderAgent
    trader = TraderAgent(weight=1.2)
    trader_result = trader.analyze(data_pack, market)
    
    # ---- Step 4: 风控师审核 ----
    from core.agents.risk_manager_agent import RiskManagerAgent
    risk_mgr = RiskManagerAgent(weight=1.0)
    risk_result = risk_mgr.analyze(trader_result, data_pack, portfolio)
    
    # ---- Step 5: 汇总 ----
    summary = _build_trade_summary(analysis, trader_result, risk_result)
    
    logger.info(f"🎯 交易管线完成: {trader_result.get('action')}")
    
    return {
        "timestamp": analysis["timestamp"],
        "market": market,
        "symbol": data_pack.get("symbol", symbol),
        "price": analysis.get("price", 0),
        "analysis": {
            "signal": analysis["signal"],
            "net_score": analysis["net_score"],
            "buy_score": analysis["buy_score"],
            "sell_score": analysis["sell_score"],
            "agent_count": analysis["agent_count"],
            "analyst_votes": analysis["analyst_votes"],
            "indicators_snapshot": analysis["indicators_snapshot"],
        },
        "trader": trader_result,
        "risk_manager": risk_result,
        "summary": summary,
    }


def _build_trade_summary(analysis: dict, trader: dict, risk: dict) -> dict:
    """构建交易汇总"""
    action = trader.get("action", "HOLD")
    
    if action == "HOLD":
        return {
            "title": "🔵 观望",
            "action": "HOLD",
            "detail": "操盘手判断当前不宜操作，继续观察",
            "signal_strength": f"净得分: {analysis.get('net_score', 0):.4f}",
        }
    
    is_long = action == "LONG"
    direction = "做多" if is_long else "做空"
    direction_icon = "🟢" if is_long else "🔴"
    
    stops = risk.get("stops", {})
    metrics = risk.get("risk_metrics", {})
    
    sl = stops.get("final_stop_loss", {}).get("price", "N/A")
    tp = stops.get("take_profit", {}).get("price", "N/A")
    
    return {
        "title": f"{direction_icon} {direction}",
        "action": action,
        "detail": trader.get("reason", ""),
        "entry_price": trader.get("entry_price", 0),
        "position_size": trader.get("size", 0),
        "stop_loss": sl,
        "take_profit": tp,
        "risk_amount": metrics.get("risk_amount", 0),
        "risk_pct": metrics.get("risk_pct_account", 0),
        "risk_reward": metrics.get("risk_reward_ratio", 0),
        "signal_strength": f"净得分: {analysis.get('net_score', 0):.4f}",
        "confidence": trader.get("confidence", 0),
    }


# =========================================================================
# 对外暴露
# =========================================================================

__all__ = ["run_ai_pipeline", "run_trade_pipeline",
           "_prepare_gold_data", "_prepare_stock_data"]
