"""
皮卡斯 API - FastAPI Serverless (Vercel)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from main import run_analysis

app = FastAPI(title="皮卡斯 Picas API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/analyze")
def analyze(market: str = Query("gold"), symbol: str = Query(None)):
    try:
        result = run_analysis(market=market, symbol=symbol)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/markets")
def markets():
    from data.fetcher import US_STOCK_MAP, A_STOCK_MAP
    from core.gold_analysts import get_gold_analysts
    from core.analysts import get_all_analysts
    return {
        "markets": {
            "gold": {
                "name": "黄金期货",
                "data": "4H+日线多周期",
                "analysts": len(get_gold_analysts()),
                "analyst_names": [a.name for a in get_gold_analysts()],
            },
            "us": {
                "name": "美股",
                "data": "日线",
                "analysts": len(get_all_analysts()),
                "analyst_names": [a.name for a in get_all_analysts()],
            },
            "a": {
                "name": "A股",
                "data": "日线",
                "analysts": len(get_all_analysts()),
                "analyst_names": [a.name for a in get_all_analysts()],
            },
        },
        "us_presets": {k: v for k, v in US_STOCK_MAP.items()},
        "a_presets": {k: f"{v[1]}{v[0]}" for k, v in A_STOCK_MAP.items()},
    }


@app.get("/health")
def health():
    return {"status": "ok"}
