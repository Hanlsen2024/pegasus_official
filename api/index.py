"""
皮卡斯 API - FastAPI Serverless (Vercel)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from main import run_analysis

app = FastAPI(title="皮卡斯 Picas API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- 内嵌前端 HTML ----------
FRONTEND = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>皮卡斯 Picas — 智能交易信号分析</title>
<style>
  :root {
    --bg: #0f0f1a; --card: #1a1a2e; --card2: #16213e;
    --accent: #e94560; --accent2: #0f3460; --gold: #f0a500;
    --green: #00c853; --red: #ff1744; --text: #e0e0e0;
    --muted: #888; --border: #2a2a4a; --radius: 12px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:var(--bg); color:var(--text); min-height:100vh; display:flex; flex-direction:column; align-items:center; }
  .header { text-align:center; padding:40px 20px 20px; }
  .header h1 { font-size:2.8rem; font-weight:900;
    background:linear-gradient(135deg,var(--accent),var(--gold));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:2px; }
  .header .sub { color:var(--muted); margin-top:6px; font-size:0.95rem; }
  .container { width:100%; max-width:640px; padding:0 20px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:28px; margin-bottom:20px; }
  .card h3 { font-size:1rem; margin-bottom:16px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
  .market-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:20px; }
  .market-btn { background:var(--card2); border:2px solid var(--border); border-radius:var(--radius);
    padding:16px 12px; cursor:pointer; text-align:center; transition:all 0.2s; color:var(--text); font-size:0.9rem; }
  .market-btn:hover { border-color:var(--accent2); }
  .market-btn.active { border-color:var(--accent); background:rgba(233,69,96,0.1); }
  .market-btn .icon { font-size:1.6rem; display:block; margin-bottom:6px; }
  .input-group { display:flex; gap:10px; }
  .input-group input { flex:1; background:var(--card2); border:2px solid var(--border);
    border-radius:var(--radius); padding:14px 16px; color:var(--text); font-size:1rem; outline:none; transition:border 0.2s; }
  .input-group input:focus { border-color:var(--accent); }
  .input-group input::placeholder { color:#555; }
  .preset-hint { margin-top:8px; font-size:0.78rem; color:var(--muted); }
  .preset-hint span { color:var(--gold); cursor:pointer; text-decoration:underline; margin:0 3px; }
  .btn-analyze { width:100%; background:linear-gradient(135deg,var(--accent),#c23152);
    color:#fff; border:none; border-radius:var(--radius); padding:16px; font-size:1.1rem;
    font-weight:700; cursor:pointer; letter-spacing:1px; transition:opacity 0.2s,transform 0.1s; }
  .btn-analyze:hover { opacity:0.9; } .btn-analyze:active { transform:scale(0.98); }
  .btn-analyze:disabled { opacity:0.4; cursor:not-allowed; }
  .loading { display:none; text-align:center; padding:20px; color:var(--muted); }
  .spinner { display:inline-block; width:32px; height:32px; border:3px solid var(--border);
    border-top-color:var(--accent); border-radius:50%; animation:spin 0.7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .result { display:none; margin-top:20px; }
  .signal-badge { display:inline-block; padding:8px 24px; border-radius:50px;
    font-size:1.4rem; font-weight:900; letter-spacing:2px; margin-bottom:16px; }
  .signal-BUY  { background:rgba(0,200,83,0.15); color:var(--green); border:2px solid var(--green); }
  .signal-SELL { background:rgba(255,23,68,0.15); color:var(--red); border:2px solid var(--red); }
  .signal-HOLD { background:rgba(240,165,0,0.15); color:var(--gold); border:2px solid var(--gold); }
  .scores { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }
  .score-item { background:var(--card2); border-radius:var(--radius); padding:14px; text-align:center; }
  .score-item .val { font-size:1.6rem; font-weight:800; }
  .score-item .lbl { font-size:0.75rem; color:var(--muted); margin-top:4px; }
  .val.buy { color:var(--green); } .val.sell { color:var(--red); } .val.net { color:var(--gold); }
  .analyst-list { display:grid; gap:8px; }
  .analyst-item { display:flex; align-items:center; justify-content:space-between;
    background:var(--card2); border-radius:var(--radius); padding:12px 16px; border-left:4px solid var(--border); }
  .analyst-item.buy { border-left-color:var(--green); }
  .analyst-item.sell { border-left-color:var(--red); }
  .analyst-item.hold { border-left-color:var(--gold); }
  .analyst-name { font-weight:600; }
  .analyst-desc { font-size:0.78rem; color:var(--muted); }
  .analyst-signal { font-size:0.8rem; font-weight:700; padding:4px 12px; border-radius:20px; }
  .analyst-signal.buy { color:var(--green); background:rgba(0,200,83,0.1); }
  .analyst-signal.sell { color:var(--red); background:rgba(255,23,68,0.1); }
  .analyst-signal.hold { color:var(--gold); background:rgba(240,165,0,0.1); }
  .symbol-label { font-size:0.85rem; color:var(--muted); margin-bottom:10px; }
  .error-msg { display:none; background:rgba(255,23,68,0.1); border:1px solid var(--red);
    color:var(--red); border-radius:var(--radius); padding:16px; margin-top:12px; font-size:0.9rem; }
  .footer { padding:30px; color:var(--muted); font-size:0.78rem; text-align:center; }
  @media (max-width:480px) {
    .header h1 { font-size:2rem; }
    .market-grid { grid-template-columns:repeat(3,1fr); gap:6px; }
    .market-btn { padding:12px 8px; font-size:0.78rem; }
    .scores { grid-template-columns:repeat(3,1fr); gap:8px; }
  }
</style>
</head>
<body>
<div class="header">
  <h1>皮卡斯 PICAS</h1>
  <p class="sub">多市场 · 多分析师 · 加权投票 · 智能交易信号</p>
</div>
<div class="container">
  <div class="card">
    <h3>📍 选择市场</h3>
    <div class="market-grid" id="marketGrid">
      <button class="market-btn active" data-market="gold"><span class="icon">🥇</span>黄金期货</button>
      <button class="market-btn" data-market="us"><span class="icon">🇺🇸</span>美股</button>
      <button class="market-btn" data-market="a"><span class="icon">🇨🇳</span>A股</button>
    </div>
    <div class="input-group">
      <input type="text" id="symbolInput" placeholder="输入代码 (如 AAPL / 600519 / 贵州茅台)">
    </div>
    <div class="preset-hint" id="presetHint">常用标的：<span onclick="fillSymbol('GC=F')">黄金期货</span></div>
  </div>
  <button class="btn-analyze" id="btnAnalyze" onclick="run()">⚡ 开始分析</button>
  <div class="loading" id="loading"><div class="spinner"></div><p style="margin-top:12px">分析师正在投票中...</p></div>
  <div class="error-msg" id="error"></div>
  <div class="result" id="result">
    <div class="card">
      <div class="symbol-label" id="symbolLabel"></div>
      <div class="signal-badge" id="signalBadge"></div>
      <div class="scores">
        <div class="score-item"><div class="val buy" id="buyScore">-</div><div class="lbl">买入得分</div></div>
        <div class="score-item"><div class="val sell" id="sellScore">-</div><div class="lbl">卖出得分</div></div>
        <div class="score-item"><div class="val net" id="netScore">-</div><div class="lbl">净得分</div></div>
      </div>
      <h3>🎯 分析师投票</h3>
      <div class="analyst-list" id="analystList"></div>
    </div>
  </div>
</div>
<div class="footer">皮卡斯 Picas v1.0.0 &nbsp;|&nbsp; 数据仅供参考，不构成投资建议</div>
<script>
let currentMarket='gold';
document.querySelectorAll('.market-btn').forEach(btn=>{btn.addEventListener('click',()=>{document.querySelectorAll('.market-btn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');currentMarket=btn.dataset.market;updatePresets();document.getElementById('symbolInput').value='';hideResult();})});
const presets={gold:[["GC=F","黄金期货"],["AU0","上期所黄金"]],us:[["AAPL","苹果"],["TSLA","特斯拉"],["MSFT","微软"],["NVDA","英伟达"],["SPY","标普500"],["QQQ","纳指100"],["AMZN","亚马逊"],["GOOGL","谷歌"],["META","Meta"],["AMD","AMD"]],a:[["600519","贵州茅台"],["000858","五粮液"],["300750","宁德时代"],["000001","平安银行"],["600036","招商银行"],["002594","比亚迪"],["601318","中国平安"],["600276","恒瑞医药"]]};
function updatePresets(){let h=document.getElementById('presetHint'),i=presets[currentMarket]||[];h.innerHTML='常用标的：'+i.map(p=>`<span onclick="fillSymbol('`+p[0]+`')">`+p[1]+'</span>').join(' ')}
function fillSymbol(s){document.getElementById('symbolInput').value=s}
function hideResult(){document.getElementById('result').style.display='none';document.getElementById('error').style.display='none'}
document.getElementById('symbolInput').addEventListener('keydown',e=>{if(e.key==='Enter')run()});
async function run(){let s=document.getElementById('symbolInput').value.trim(),b=document.getElementById('btnAnalyze'),l=document.getElementById('loading'),e=document.getElementById('error');hideResult();b.disabled=true;l.style.display='block';try{let u='/analyze?market='+currentMarket;if(s)u+='&symbol='+encodeURIComponent(s);let r=await fetch(u);if(!r.ok){let j=await r.json();throw new Error(j.error||'HTTP '+r.status)}let d=await r.json();renderResult(d,s)}catch(err){e.textContent='❌ '+err.message;e.style.display='block'}finally{b.disabled=false;l.style.display='none'}}
function renderResult(d,s){document.getElementById('result').style.display='block';let n={gold:'黄金期货',us:'美股',a:'A股'};document.getElementById('symbolLabel').textContent=(n[currentMarket]||currentMarket)+' · '+(d.symbol||s||'-');let b=document.getElementById('signalBadge');b.textContent=d.signal||'-';b.className='signal-badge signal-'+(d.signal||'HOLD');document.getElementById('buyScore').textContent=d.buy_score?.toFixed(2)||'-';document.getElementById('sellScore').textContent=d.sell_score?.toFixed(2)||'-';document.getElementById('netScore').textContent=d.net_score?.toFixed(2)||'-';let l=document.getElementById('analystList');l.innerHTML='';if(d.analyst_votes)d.analyst_votes.forEach(v=>{let c=(v.signal||'').toLowerCase();l.innerHTML+=`<div class="analyst-item `+c+`"><div><div class="analyst-name">`+v.name+`</div><div class="analyst-desc">`+(v.reason||'')+`</div></div><span class="analyst-signal `+c+`">`+v.signal+`</span></div>`})}
updatePresets();
</script>
</body>
</html>"""


# ---------- 路由 ----------

@app.get("/", response_class=HTMLResponse)
def root():
    return FRONTEND


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/markets")
def markets():
    from data.fetcher import US_STOCK_MAP, A_STOCK_MAP
    return {
        "markets": {"gold":"黄金期货 (COMEX/上期所)","us":"美股 (yfinance)","a":"A股 (akshare)"},
        "us_presets": {k:v for k,v in US_STOCK_MAP.items()},
        "a_presets": {k:f"{v[1]}{v[0]}" for k,v in A_STOCK_MAP.items()},
    }


@app.get("/analyze")
def analyze(
    market: str = Query("gold", description="gold / us / a"),
    symbol: str = Query(None, description="股票代码"),
):
    try:
        result = run_analysis(market=market, symbol=symbol)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
