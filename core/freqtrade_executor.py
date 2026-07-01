"""
Freqtrade 被动执行模式 — 通过 Freqtrade REST API 执行交易指令

对标 Brale Fast Loop 中的执行层：
- 开仓: POST /api/v1/forcebuy
- 平仓: POST /api/v1/forcesell
- 止损: POST /api/v1/forcesell (emergency)
- 状态查询: GET /api/v1/status
- 持仓查询: GET /api/v1/trades

工作模式：
- dry_run=True: 模拟交易 (Freqtrade 内部模拟)
- dry_run=False: 实盘交易 (对接交易所)
"""
import logging
import time
from typing import Optional, Dict, Any
from urllib.parse import urljoin

logger = logging.getLogger("picas.freqtrade")

# 尝试导入 requests，无则降级
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests 未安装，Freqtrade 执行器仅支持回调模式")


class FreqtradeExecutor:
    """
    Freqtrade REST API 执行器
    
    用法:
        executor = FreqtradeExecutor(base_url="http://localhost:8080", api_token="xxx")
        executor.connect()
        executor.buy("BTC/USDT", amount=100)
        executor.sell("BTC/USDT", reason="take_profit")
    """

    def __init__(self, base_url: str = "http://localhost:8080",
                 api_token: str = "", dry_run: bool = True,
                 stake_currency: str = "USDT",
                 stake_amount: float = 100,
                 max_open_trades: int = 3,
                 timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.dry_run = dry_run
        self.stake_currency = stake_currency
        self.stake_amount = stake_amount
        self.max_open_trades = max_open_trades
        self.timeout = timeout
        self.connected = False
        self._session = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """测试 Freqtrade API 连接"""
        if not HAS_REQUESTS:
            logger.error("requests 未安装，无法连接 Freqtrade")
            return False

        try:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            })
            resp = self._session.get(
                urljoin(self.base_url, "/api/v1/ping"),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                self.connected = True
                logger.info(f"✅ Freqtrade 连接成功: {self.base_url}")
                return True
            else:
                logger.warning(f"Freqtrade 连接失败: HTTP {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"Freqtrade 连接异常: {e}")
            return False

    def get_status(self) -> Optional[dict]:
        """获取 Freqtrade 运行状态"""
        if not self.connected or not self._session:
            return None
        try:
            resp = self._session.get(
                urljoin(self.base_url, "/api/v1/show_config"),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
        return None

    # ------------------------------------------------------------------
    # 交易指令
    # ------------------------------------------------------------------

    def buy(self, pair: str, amount: float = None,
            price: float = None) -> Optional[dict]:
        """
        开仓买入
        
        Args:
            pair: 交易对 (如 "BTC/USDT")
            amount: 买入金额 (默认使用 stake_amount)
            price: 限价单价格 (None=市价)
        
        Returns:
            dict: {"trade_id": ..., "order_id": ...} 或 None
        """
        if not self.connected or not self._session:
            logger.error("Freqtrade 未连接")
            return None

        if self.dry_run:
            logger.info(f"🏴 [DRY RUN] 模拟买入 {pair} 金额{amount or self.stake_amount}")

        payload = {
            "pair": pair,
            "price": price,
        }
        if amount is not None:
            payload["stake_amount"] = amount

        try:
            resp = self._session.post(
                urljoin(self.base_url, "/api/v1/forcebuy"),
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"✅ 开仓成功: {pair} trade_id={result.get('trade_id')}")
                return result
            else:
                logger.error(f"开仓失败: {pair} HTTP {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            logger.error(f"开仓异常: {pair} {e}")
            return None

    def sell(self, trade_id: int, reason: str = "manual",
             amount: float = None) -> Optional[dict]:
        """
        平仓卖出
        
        Args:
            trade_id: Freqtrade 交易ID
            reason: 平仓原因 (stop_loss / take_profit / emergency / manual)
            amount: 平仓比例 (None=全部)
        
        Returns:
            dict 或 None
        """
        if not self.connected or not self._session:
            logger.error("Freqtrade 未连接")
            return None

        if self.dry_run:
            logger.info(f"🏴 [DRY RUN] 模拟平仓 trade_id={trade_id} reason={reason}")

        payload = {
            "tradeid": str(trade_id),
            "ordertype": "market",
        }
        if amount is not None:
            payload["amount"] = amount

        try:
            # 紧急平仓使用 forcesell
            endpoint = "/api/v1/forcesell"
            resp = self._session.post(
                urljoin(self.base_url, endpoint),
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"✅ 平仓成功: trade_id={trade_id} reason={reason}")
                return result
            else:
                logger.error(f"平仓失败: trade_id={trade_id} HTTP {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"平仓异常: trade_id={trade_id} {e}")
            return None

    def emergency_close_all(self) -> list:
        """紧急平仓所有持仓"""
        results = []
        trades = self.get_open_trades()
        for trade in trades:
            tid = trade.get("trade_id")
            if tid:
                result = self.sell(tid, reason="emergency")
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list:
        """获取当前开放持仓"""
        if not self.connected or not self._session:
            return []
        try:
            resp = self._session.get(
                urljoin(self.base_url, "/api/v1/trades"),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                trades = resp.json().get("trades", [])
                return [t for t in trades if t.get("is_open")]
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
        return []

    def get_balance(self) -> Optional[dict]:
        """获取账户余额"""
        if not self.connected or not self._session:
            return None
        try:
            resp = self._session.get(
                urljoin(self.base_url, "/api/v1/balance"),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"查询余额失败: {e}")
        return None

    def get_profit(self) -> Optional[dict]:
        """获取盈亏统计"""
        if not self.connected or not self._session:
            return None
        try:
            resp = self._session.get(
                urljoin(self.base_url, "/api/v1/profit"),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"查询盈亏失败: {e}")
        return None

    # ------------------------------------------------------------------
    # Fast Loop 执行回调 (与 DualLoopScheduler 对接)
    # ------------------------------------------------------------------

    def on_execution_event(self, event: dict):
        """
        Fast Loop 触发的执行回调
        
        event 结构:
        {
            "type": "stop_loss"|"take_profit"|"ladder_tp"|"hard_stop"|"trailing_sl_update",
            "plan": {...},
            "price": 123.45,
            "message": "xxx",
        }
        """
        event_type = event.get("type", "")
        plan = event.get("plan", {})
        price = event.get("price", 0)
        symbol = plan.get("symbol", "N/A")
        message = event.get("message", "")

        logger.info(f"📡 Freqtrade 执行: {message}")

        # 转换为 Freqtrade 交易对格式
        pair = self._symbol_to_pair(symbol)

        if event_type == "hard_stop":
            # 紧急平仓
            self._close_pair_position(pair, "hard_stop")

        elif event_type == "stop_loss":
            self._close_pair_position(pair, "stop_loss")

        elif event_type == "take_profit":
            self._close_pair_position(pair, "take_profit")

        elif event_type == "ladder_tp":
            close_ratio = float(event.get("close_ratio", "100%").rstrip("%")) / 100
            self._close_pair_position(pair, "ladder_tp", ratio=close_ratio)

        elif event_type == "trailing_sl_update":
            new_sl = event.get("new_sl")
            old_sl = event.get("old_sl")
            logger.info(f"🔄 {symbol} 追踪止损更新: {old_sl} → {new_sl}")
            # Freqtrade 内部有追踪止损机制，这里仅记录

    def _symbol_to_pair(self, symbol: str) -> str:
        """将 symbol 转为 Freqtrade 标准交易对格式"""
        # 如 BTC-USD → BTC/USDT
        if "-" in symbol:
            base = symbol.split("-")[0]
            return f"{base}/{self.stake_currency}"
        return f"{symbol}/{self.stake_currency}"

    def _close_pair_position(self, pair: str, reason: str, ratio: float = 1.0):
        """关闭指定交易对的持仓"""
        trades = self.get_open_trades()
        for trade in trades:
            if trade.get("pair") == pair:
                tid = trade.get("trade_id")
                if ratio >= 1.0:
                    self.sell(tid, reason=reason)
                else:
                    # 部分平仓
                    stake = trade.get("stake_amount", 0)
                    partial = stake * ratio
                    self.sell(tid, reason=reason, amount=partial)


# ---------------------------------------------------------------------------
# 工厂方法
# ---------------------------------------------------------------------------

def create_freqtrade_executor(market: str = "default") -> Optional[FreqtradeExecutor]:
    """
    从 config.yaml 创建 Freqtrade 执行器
    
    Args:
        market: 市场类型 (仅用于日志区分)
    """
    from config.loader import load_config
    cfg = load_config().get("freqtrade", {})

    if not cfg.get("enabled", False):
        logger.info(f"Freqtrade 未启用 (market={market})")
        return None

    executor = FreqtradeExecutor(
        base_url=cfg.get("base_url", "http://localhost:8080"),
        api_token=cfg.get("api_token", ""),
        dry_run=cfg.get("dry_run", True),
        stake_currency=cfg.get("stake_currency", "USDT"),
        stake_amount=cfg.get("stake_amount", 100),
        max_open_trades=cfg.get("max_open_trades", 3),
        timeout=cfg.get("timeout", 10),
    )

    if HAS_REQUESTS:
        executor.connect()

    return executor


def get_or_create_executor(market: str = "default") -> Optional[FreqtradeExecutor]:
    """单例获取 Freqtrade 执行器"""
    global _freqtrade_executor
    if _freqtrade_executor is None:
        _freqtrade_executor = create_freqtrade_executor(market)
    return _freqtrade_executor


_freqtrade_executor: Optional[FreqtradeExecutor] = None
