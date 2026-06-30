"""
皮卡斯 API - FastAPI Serverless (Vercel 部署入口)
"""

import sys
import os

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from main import run_analysis

app = FastAPI(
    title="皮卡斯 Picas API",
    description="多市场多分析师交易信号投票系统",
    version="1.0.0",
)

# CORS 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 前端 HTML 页面（缓存内容，避免每次读取文件）
_FRONTEND_HTML = None


def _load_html():
    global _FRONTEND_HTML
    if _FRONTEND_HTML is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_path = os.path.join(base_dir, "static", "index.html")
        try:
            with open(static_path, "r", encoding="utf-8") as f:
                _FRONTEND_HTML = f.read()
        except FileNotFoundError:
            _FRONTEND_HTML = ""  # fallback
    return _FRONTEND_HTML


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    """前端页面"""
    html = _load_html()
    if html:
        return html
    # fallback: JSON API 信息
    return {
        "service": "皮卡斯 Picas",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "GET - 执行分析 ?market=gold/us/a&symbol=代码",
            "/markets": "GET - 查看支持的市场列表",
            "/health": "GET - 健康检查",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/markets")
def markets():
    """查看支持的市场和预设标的"""
    from data.fetcher import US_STOCK_MAP, A_STOCK_MAP

    return {
        "markets": {
            "gold": "黄金期货 (COMEX/上期所)",
            "us": "美股 (yfinance)",
            "a": "A股 (akshare)",
        },
        "us_presets": {k: v for k, v in US_STOCK_MAP.items()},
        "a_presets": {k: f"{v[1]}{v[0]}" for k, v in A_STOCK_MAP.items()},
    }


@app.get("/analyze")
def analyze(
    market: str = Query("gold", description="市场: gold / us / a"),
    symbol: str = Query(None, description="股票代码 (美股如AAPL, A股如600519)"),
):
    """
    执行一次完整分析

    示例:
      /analyze?market=gold
      /analyze?market=us&symbol=AAPL
      /analyze?market=a&symbol=600519
      /analyze?market=a&symbol=贵州茅台
    """
    try:
        result = run_analysis(market=market, symbol=symbol)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
