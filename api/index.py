"""
皮卡斯 2.0 API — FastAPI Serverless
新增: 操盘手 + 风控师 完整交易管线
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(title="皮卡斯 Picas 2.0 API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def index():
    """前端页面"""
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/analyze")
def analyze(market: str = Query("gold"), symbol: str = Query(None)):
    """核心分析接口 — AI Agent 驱动"""
    try:
        from core.pipeline import run_ai_pipeline
        result = run_ai_pipeline(market=market, symbol=symbol)
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/trade")
@app.get("/trade")
def trade(
    market: str = Query("gold"),
    symbol: str = Query(None),
    equity: float = Query(10000),
    risk_per_trade: float = Query(0.02),
    position_type: Optional[str] = Query(None, alias="pos_type"),
    position_entry: Optional[float] = Query(None, alias="pos_entry"),
    position_size: Optional[float] = Query(None, alias="pos_size"),
    position_sl: Optional[float] = Query(None, alias="pos_sl"),
):
    """
    完整交易管线: 3分析师 → 操盘手 → 风控师
    
    用法: GET /trade?market=gold&equity=10000&risk_per_trade=0.02
    
    有持仓时:
      /trade?market=gold&equity=10000&pos_type=long&pos_entry=870&pos_size=0.1&pos_sl=875
    """
    try:
        portfolio = {
            "equity": equity,
            "risk_per_trade": risk_per_trade,
            "risk_reward_ratio": 2.0,
            "position": None,
        }
        
        if position_type and position_type.lower() in ("long", "short"):
            portfolio["position"] = {
                "type": position_type.lower(),
                "entry_price": position_entry or 0,
                "size": position_size or 0,
                "stop_loss": position_sl or 0,
            }
        
        from core.pipeline import run_trade_pipeline
        result = run_trade_pipeline(market=market, symbol=symbol, portfolio=portfolio)
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/markets")
def markets():
    """市场与 Agent 信息"""
    from config.loader import get_market_config
    markets_info = {}
    for m in ["gold", "us_stock", "a_stock"]:
        cfg = get_market_config(m)
        markets_info[m] = {
            "name": cfg.get("name", m),
            "timeframes": cfg.get("timeframes", []),
            "agents": cfg.get("agents", []),
            "presets": cfg.get("presets", {}),
        }
    return {
        "service": "皮卡斯 Picas 2.0",
        "architecture": "AI驱动的多Agent量化信号引擎 (对标Brale Dual-Loop)",
        "agents": [
            "Indicator Agent(指标共振)",
            "Pattern Agent(形态识别)",
            "Trend Agent(趋势方向)",
            "Trader Agent(操盘手)",
            "Risk Manager(风控师)",
        ],
        "llm": "OpenAI兼容协议 (GPT/Claude/DeepSeek)",
        "markets": markets_info,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
