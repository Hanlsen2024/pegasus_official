"""
风控师 Agent (Risk Manager) — 规则引擎（非LLM，确保可靠性）
提供：初始止损、追踪止损、保本止损、硬止损、止盈

对标 Brale Fast Loop 中的风控层
"""
import logging

logger = logging.getLogger("picas.risk_manager")


class RiskManagerAgent:
    """
    风控师 — 规则驱动，纯计算
    
    止损体系（从紧到松）：
    1. 初始止损：开仓时必设，基于 ATR
    2. 追踪止损：盈利后动态上移，锁定利润
    3. 保本止损：盈利达到阈值后移到入场价
    4. 硬止损：绝不允许超过此价格
    
    所用参数均从 config.yaml risk 段读取。
    """
    
    def __init__(self, weight: float = 1.0):
        self.name = "🛡️ 风控师"
        self.weight = weight
    
    def analyze(self, trader_decision: dict, data_pack: dict, portfolio: dict) -> dict:
        """
        根据操盘手决策和持仓状态，计算风控参数
        
        Args:
            trader_decision: 操盘手的输出 {"action": "LONG"|"SHORT"|"CLOSE"|"HOLD", "size": ..., "entry_price": ...}
            data_pack: 市场数据 {"price": ..., "atr": ..., "atr_pct": ..., "symbol": ...}
            portfolio: 账户状态 {"equity": ..., "risk_per_trade": ..., "position": ...}
        
        Returns:
            dict: 风控参数
        """
        action = trader_decision.get("action", "HOLD")
        entry_price = trader_decision.get("entry_price", data_pack.get("price", 0))
        size = trader_decision.get("size", 0)
        
        # 获取配置参数
        from config.loader import get_risk_config
        risk_cfg = get_risk_config()
        
        # 计算 ATR 点数
        atr = data_pack.get("atr", 0)
        atr_pct = data_pack.get("atr_pct", 1.5) or 1.5
        
        equity = portfolio.get("equity", 10000)
        risk_pct = portfolio.get("risk_per_trade", 0.02)
        max_risk_amount = equity * risk_pct
        
        if action == "HOLD":
            # 有持仓则检查是否需要调整
            position = portfolio.get("position")
            if position and position.get("type"):
                return self._update_existing_position(position, data_pack, portfolio)
            return {
                "action": "HOLD",
                "message": "无操作，风控系统待命",
                "stops": {}
            }
        
        if action == "CLOSE":
            return {
                "action": "CLOSE",
                "message": "操盘手指令平仓，执行中...",
                "stops": {}
            }
        
        # ---- 开仓风控设置 ----
        is_long = action == "LONG"
        
        # 1. 初始止损（ATR 倍数）
        atr_mult = risk_cfg.get("atr_trailing_mult", 2.0)
        stop_distance = atr * atr_mult
        if is_long:
            initial_sl = round(entry_price - stop_distance, 2)
        else:
            initial_sl = round(entry_price + stop_distance, 2)
        
        # 2. 硬止损（不得超过最大亏损金额）
        if size > 0 and stop_distance > 0:
            max_loss_points = max_risk_amount / size
            max_sl_distance = min(stop_distance, max_loss_points)
            if is_long:
                hard_sl = round(entry_price - max_sl_distance, 2)
            else:
                hard_sl = round(entry_price + max_sl_distance, 2)
        else:
            hard_sl = initial_sl
        
        # 3. 保本止损触发条件
        breakeven_trigger = atr * 1.0  # 盈利超过 1x ATR 移至成本
        
        # 4. 追踪止损参数
        trail_trigger = atr * 1.5   # 盈利超过 1.5x ATR 开始追踪
        trail_distance = atr * atr_mult * 0.5  # 追踪距离 = 初始止损的一半
        
        # 5. 止盈（基于风险回报比）
        rr_ratio = portfolio.get("risk_reward_ratio", 2.0)
        tp_distance = stop_distance * rr_ratio
        if is_long:
            take_profit = round(entry_price + tp_distance, 2)
        else:
            take_profit = round(entry_price - tp_distance, 2)
        
        # 6. 止损百分比
        sl_pct = risk_cfg.get("default_sl_pct", 3.0)
        if is_long:
            sl_price_from_pct = round(entry_price * (1 - sl_pct / 100), 2)
        else:
            sl_price_from_pct = round(entry_price * (1 + sl_pct / 100), 2)
        
        # 取较紧的止损
        if is_long:
            final_sl = max(initial_sl, sl_price_from_pct)  # 做多取较近的止损
        else:
            final_sl = min(initial_sl, sl_price_from_pct)  # 做空取较近的止损
        
        # 阶梯止盈
        ladder = risk_cfg.get("ladder_tp", [])
        ladder_plan = []
        for step in ladder:
            pct = step.get("pct", 5)
            ratio = step.get("close_ratio", 0.3)
            if is_long:
                target = round(entry_price * (1 + pct / 100), 2)
            else:
                target = round(entry_price * (1 - pct / 100), 2)
            ladder_plan.append({
                "profit_pct": pct,
                "target_price": target,
                "close_ratio": f"{ratio:.0%}",
                "close_size": round(size * ratio, 2),
            })
        
        # 风险金额
        if size > 0:
            risk_amount = round(abs(entry_price - final_sl) * size, 2)
        else:
            risk_amount = 0
        reward_amount = round(abs(entry_price - take_profit) * size, 2) if size > 0 else 0
        
        # 7. 规则检查引擎
        rules_check = self._check_rules(
            entry_price=entry_price,
            final_sl=final_sl,
            hard_sl=hard_sl,
            take_profit=take_profit,
            max_risk_amount=max_risk_amount,
            risk_amount=risk_amount,
            equity=equity,
            is_long=is_long,
        )
        
        result = {
            "action": action,
            "stops": {
                "initial_stop_loss": {
                    "price": initial_sl,
                    "method": f"ATR×{atr_mult:.1f}",
                    "distance": round(stop_distance, 2),
                },
                "final_stop_loss": {
                    "price": final_sl,
                    "method": "取ATR止损与百分比止损较紧者",
                    "pct_from_entry": round(abs(entry_price - final_sl) / entry_price * 100, 2),
                },
                "hard_stop": {
                    "price": hard_sl,
                    "method": "绝对最大亏损",
                    "max_loss_amount": round(max_risk_amount, 2),
                },
                "trailing_stop": {
                    "enabled": True,
                    "trigger_distance": round(trail_trigger, 2),
                    "trail_distance": round(trail_distance, 2),
                    "method": f"盈利>{trail_trigger:.1f}点后启动，追踪距离{trail_distance:.1f}点",
                },
                "breakeven_stop": {
                    "enabled": True,
                    "trigger_distance": round(breakeven_trigger, 2),
                    "method": f"盈利>{breakeven_trigger:.1f}点后SL移至入场价",
                },
                "take_profit": {
                    "price": take_profit,
                    "risk_reward_ratio": rr_ratio,
                    "distance": round(tp_distance, 2),
                },
                "ladder_tp": ladder_plan,
            },
            "risk_metrics": {
                "risk_amount": risk_amount,
                "reward_amount": reward_amount,
                "risk_pct_account": round(risk_amount / equity * 100, 2) if equity > 0 else 0,
                "risk_reward_ratio": round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0,
            },
            "rules_check": rules_check,
            "message": self._build_message(is_long, entry_price, final_sl, take_profit, risk_amount, risk_pct, rules_check),
        }
        
        logger.info(f"[风控师] {action} 入场{entry_price} 止损{final_sl} 止盈{take_profit} 风险${risk_amount}")
        return result
    
    def _update_existing_position(self, position: dict, data_pack: dict, portfolio: dict) -> dict:
        """更新已有持仓的追踪止损"""
        pos_type = position.get("type", "long")
        entry = position.get("entry_price", 0)
        current_sl = position.get("stop_loss", 0)
        price = data_pack.get("price", 0)
        atr = data_pack.get("atr", 0)
        
        if pos_type == "long":
            pnl_points = price - entry
            if pnl_points > atr * 1.0 and current_sl < entry:
                # 移至保本
                new_sl = entry
                return {
                    "action": "UPDATE_SL", 
                    "message": f"触发保本止损: SL 从 {current_sl} 移至 {new_sl}",
                    "stops": {"new_stop_loss": new_sl},
                }
            elif pnl_points > atr * 1.5:
                new_sl = max(current_sl, price - atr * 1.0)
                return {
                    "action": "UPDATE_SL",
                    "message": f"追踪止损: SL 从 {current_sl} 上移至 {new_sl}",
                    "stops": {"new_stop_loss": round(new_sl, 2)},
                }
        else:
            pnl_points = entry - price
            if pnl_points > atr * 1.0 and current_sl > entry:
                new_sl = entry
                return {
                    "action": "UPDATE_SL",
                    "message": f"触发保本止损: SL 从 {current_sl} 移至 {new_sl}",
                    "stops": {"new_stop_loss": new_sl},
                }
            elif pnl_points > atr * 1.5:
                new_sl = min(current_sl, price + atr * 1.0)
                return {
                    "action": "UPDATE_SL",
                    "message": f"追踪止损: SL 从 {current_sl} 下移至 {new_sl}",
                    "stops": {"new_stop_loss": round(new_sl, 2)},
                }
        
        return {"action": "HOLD", "message": "持仓状态正常，风控系统监控中", "stops": {}}
    
    def _check_rules(self, entry_price, final_sl, hard_sl, take_profit,
                     max_risk_amount, risk_amount, equity, is_long) -> list:
        """风控规则检查"""
        checks = []
        
        # 规则1：不能超过硬止损
        if is_long and final_sl < hard_sl:
            checks.append({"rule": "硬止损约束", "status": "fail", 
                          "msg": f"止损{final_sl}越过硬止损{hard_sl}", "fix": f"使用硬止损{hard_sl}"})
        elif not is_long and final_sl > hard_sl:
            checks.append({"rule": "硬止损约束", "status": "fail",
                          "msg": f"止损{final_sl}越过硬止损{hard_sl}", "fix": f"使用硬止损{hard_sl}"})
        else:
            checks.append({"rule": "硬止损约束", "status": "pass"})
        
        # 规则2：风险是否在预算内
        if risk_amount > max_risk_amount:
            checks.append({"rule": "风险预算", "status": "warn",
                          "msg": f"风险${risk_amount}超出预算${max_risk_amount}",
                          "suggestion": "减小仓位或调整止损"})
        else:
            checks.append({"rule": "风险预算", "status": "pass",
                          "msg": f"风险${risk_amount}在预算${max_risk_amount}内"})
        
        # 规则3：风险/回报比
        if entry_price and final_sl and take_profit and risk_amount > 0:
            rr = abs(entry_price - take_profit) / abs(entry_price - final_sl)
            if rr < 1.5:
                checks.append({"rule": "风报比", "status": "warn",
                              "msg": f"风报比{rr:.1f}:1 偏低，建议≥1.5:1"})
            else:
                checks.append({"rule": "风报比", "status": "pass",
                              "msg": f"风报比{rr:.1f}:1"})
        
        # 规则4：单笔风险不超过账户5%
        if equity > 0 and risk_amount / equity > 0.05:
            checks.append({"rule": "账户风险上限", "status": "fail",
                          "msg": f"单笔风险{risk_amount/equity:.1%}超过5%上限"})
        else:
            checks.append({"rule": "账户风险上限", "status": "pass",
                          "msg": f"单笔风险{risk_amount/equity:.1%}"})
        
        return checks
    
    def _build_message(self, is_long, entry, sl, tp, risk_amount, risk_pct, checks):
        """构建风控报告摘要"""
        direction = "做多" if is_long else "做空"
        fails = [c for c in checks if c["status"] == "fail"]
        warns = [c for c in checks if c["status"] == "warn"]
        
        msg = f"风控报告：{direction}入场{entry}，止损{sl}，止盈{tp}，风险${risk_amount:.0f}"
        
        if fails:
            msg += f" ⚠️ 违规{len(fails)}项！"
        elif warns:
            msg += f" ⚡ 警告{len(warns)}项"
        else:
            msg += " ✅ 全部通过"
        
        return msg
