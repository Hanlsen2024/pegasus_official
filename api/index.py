"""
皮卡斯 2.0 API — FastAPI Serverless (Vercel)
新增: AI推理结果展示、推理过程返回、计划管理
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

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
        "agents": ["Indicator Agent(指标共振)", "Pattern Agent(形态识别)", "Trend Agent(趋势方向)"],
        "llm": "OpenAI兼容协议 (GPT/Claude/DeepSeek)",
        "markets": markets_info,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
